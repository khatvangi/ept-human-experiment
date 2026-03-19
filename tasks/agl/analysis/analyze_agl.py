#!/usr/bin/env python3
"""
AGL analysis — thin adapter into the inference module.

loads participant JSON data from the AGL experiment and feeds it
through the universal transition-detection pipeline. replaces
the old bespoke analysis script entirely.

usage:
    python tasks/agl/analysis/analyze_agl.py data/
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

import sys
import os
import json
import glob
import numpy as np
from scipy import stats

# add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from inference import detect_transitions
from inference.pipeline import detect_transitions_cohort
from inference.persistence import test_persistence_controlled


# ─────────────────────────────────────────────
# data loading: AGL JSON → canonical format
# ─────────────────────────────────────────────

def load_participants(data_dir):
    """load all participant JSON files from data directory."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    participants = []
    for f in files:
        with open(f) as fh:
            participants.append(json.load(fh))
    return participants


def participant_to_canonical(p):
    """
    convert one AGL participant's JSON to the canonical format
    expected by inference.detect_transitions().

    canonical format:
        accuracy: 1D array of trial-level accuracy (0/1)
        confidence: 1D array of confidence ratings (interpolated to trial level)
        rt: 1D array of reaction times in ms
        transfer: 1D array of transfer-phase accuracy
        metadata: dict with participant_id, condition, aha_trial, etc.
    """
    # extract learning-phase trials
    learning = [t for t in p["trials"] if t["phase"] == "learning"]
    transfer = [t for t in p["trials"] if t["phase"] == "transfer"]

    if len(learning) < 20:
        return None

    accuracy_raw = np.array([t["correct"] for t in learning], dtype=float)
    rt = np.array([t["rt_ms"] for t in learning], dtype=float)

    # window binary accuracy into a continuous signal for inference.
    # raw 0/1 per trial is too noisy for changepoint detection.
    # use rolling mean with window=10 (matches confidence probe interval).
    window = min(10, len(accuracy_raw) // 5)
    if window >= 3:
        kernel = np.ones(window) / window
        accuracy = np.convolve(accuracy_raw, kernel, mode="same")
    else:
        accuracy = accuracy_raw

    # interpolate confidence to trial level
    # (confidence is sampled every 10 trials in the AGL design)
    conf_sparse = p.get("confidence_trajectory", [])
    confidence = _interpolate_confidence(conf_sparse, len(learning))

    # transfer accuracy
    transfer_acc = None
    if len(transfer) >= 5:
        transfer_acc = np.array([t["correct"] for t in transfer], dtype=float)

    # aha trial (self-report marker)
    aha_trial = None
    if p.get("aha_events"):
        aha_trial = p["aha_events"][0]["trial_num"]

    # final learning accuracy (for persistence control)
    final_acc = float(accuracy[-20:].mean()) if len(accuracy) >= 20 else float(accuracy.mean())

    return {
        "accuracy": accuracy,
        "confidence": confidence,
        "rt": rt,
        "transfer": transfer_acc,
        "metadata": {
            "participant_id": p["participant_id"],
            "condition": p["condition"],
            "aha_trial": aha_trial,
            "has_aha": aha_trial is not None,
            "final_learning_acc": final_acc,
            "n_learning_trials": len(learning),
            "n_transfer_trials": len(transfer),
        },
    }


def _interpolate_confidence(conf_trajectory, n_trials):
    """
    interpolate sparse confidence ratings (every 10 trials) to trial level.
    uses linear interpolation between probes.
    """
    if not conf_trajectory:
        return None

    # build sparse series
    trial_nums = [c["after_trial"] for c in conf_trajectory]
    ratings = [c["rating"] for c in conf_trajectory]

    if len(trial_nums) < 2:
        return None

    # interpolate
    conf = np.interp(
        np.arange(n_trials),
        trial_nums,
        ratings,
        left=ratings[0],
        right=ratings[-1],
    )
    return conf


# ─────────────────────────────────────────────
# analysis: run inference + AGL-specific tests
# ─────────────────────────────────────────────

def analyze_cohort(participants):
    """
    run full analysis on all participants.

    analysis structure:
      S1. per-participant transition detection (via inference module)
      S2. dose-response: transition rate by grammar difficulty
      S3. persistence: transfer accuracy controlling for learning performance
      S4. convergence: aha self-report vs inference-detected changepoints
    """
    # convert to canonical format
    canonical = []
    metadata = []
    for p in participants:
        c = participant_to_canonical(p)
        if c is not None:
            canonical.append(c)
            metadata.append(c["metadata"])

    if not canonical:
        print("no valid participants after preprocessing.")
        return None

    print(f"loaded {len(canonical)} participants "
          f"({len(participants) - len(canonical)} excluded for insufficient trials)")

    # ─── S1: per-participant transition detection ───
    print("\n" + "=" * 60)
    print("S1: TRANSITION DETECTION (per participant)")
    print("=" * 60)

    cohort_result = detect_transitions_cohort(
        canonical, chance_level=0.5
    )

    summary = cohort_result["cohort_summary"]
    print(f"\n  classification summary (n={summary['n']}):")
    for label, info in sorted(summary["label_fractions"].items()):
        count = summary["label_counts"][label]
        print(f"    {label:15s}: {count:3d} ({info:.0%})")

    # per-participant detail
    for i, (r, m) in enumerate(zip(cohort_result["individual"], metadata)):
        label = r["classification"]["label"]
        conf = r["classification"]["confidence"]
        aha = "aha" if m["has_aha"] else "   "
        cond = m["condition"]
        print(f"    {m['participant_id'][:10]:10s} [{cond:6s}] {aha} → "
              f"{label:12s} ({conf})")

    # ─── S2: dose-response ───
    print("\n" + "=" * 60)
    print("S2: DOSE-RESPONSE (transition rate by difficulty)")
    print("=" * 60)

    by_condition = {}
    for r, m in zip(cohort_result["individual"], metadata):
        cond = m["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append({
            "label": r["classification"]["label"],
            "has_aha": m["has_aha"],
            "final_acc": m["final_learning_acc"],
        })

    contingency = []
    for cond in sorted(by_condition.keys()):
        entries = by_condition[cond]
        n = len(entries)
        n_abrupt = sum(1 for e in entries if e["label"] == "abrupt")
        n_aha = sum(1 for e in entries if e["has_aha"])
        mean_acc = np.mean([e["final_acc"] for e in entries])
        print(f"  {cond:8s}: n={n}, abrupt={n_abrupt}/{n} ({n_abrupt/n:.0%}), "
              f"aha={n_aha}/{n} ({n_aha/n:.0%}), final_acc={mean_acc:.3f}")
        contingency.append([n_abrupt, n - n_abrupt])

    if len(contingency) >= 2 and all(sum(row) > 0 for row in contingency):
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        print(f"\n  chi-squared (abrupt rate × condition): "
              f"chi2={chi2:.3f}, p={p_val:.4f}, dof={dof}")

    # ─── S3: persistence (controlled) ───
    print("\n" + "=" * 60)
    print("S3: PERSISTENCE (transfer accuracy, controlled)")
    print("=" * 60)

    # collect transfer data with transition labels
    transfer_data = []
    learning_perf = []
    for r, m, c in zip(cohort_result["individual"], metadata, canonical):
        if c["transfer"] is not None:
            had_transition = r["classification"]["label"] == "abrupt"
            transfer_acc = float(c["transfer"].mean())
            transfer_data.append((transfer_acc, had_transition))
            learning_perf.append(m["final_learning_acc"])

    if len(transfer_data) >= 10:
        persist = test_persistence_controlled(
            transfer_data, learning_perf, chance_level=0.5
        )
        if persist.get("sufficient_data"):
            print(f"  n_transition: {persist['n_transition']}, "
                  f"n_no_transition: {persist['n_no_transition']}")
            print(f"  learning-transfer r: {persist['learning_transfer_r']:.3f}")
            print(f"  residual effect (transition): {persist['residual_transfer_effect']:.3f}")
            print(f"  effect size d: {persist['effect_size_d']:.3f}")
            print(f"  p-value: {persist['p_value']:.4f}")
            print(f"  → {persist['interpretation']}")
        else:
            print("  insufficient data for controlled persistence test")
            persist = None
    else:
        print(f"  insufficient transfer data ({len(transfer_data)} participants)")
        persist = None

    # ─── S4: convergence — aha vs detected changepoints ───
    print("\n" + "=" * 60)
    print("S4: AHA-CHANGEPOINT ALIGNMENT")
    print("=" * 60)

    aha_cp_distances = []
    for r, m in zip(cohort_result["individual"], metadata):
        if not m["has_aha"]:
            continue
        aha_trial = m["aha_trial"]
        cps = r["changepoints"]["changepoints"]
        if cps:
            # distance from aha to nearest changepoint
            nearest = min(abs(aha_trial - cp) for cp in cps)
            aha_cp_distances.append(nearest)
            print(f"    {m['participant_id'][:10]:10s}: aha={aha_trial}, "
                  f"changepoints={cps}, nearest={nearest}")

    if aha_cp_distances:
        d = np.array(aha_cp_distances)
        print(f"\n  n aha participants with changepoints: {len(d)}")
        print(f"  mean |aha - nearest CP|: {d.mean():.1f} trials")
        print(f"  median: {np.median(d):.1f}")
        print(f"  within 5 trials: {(d <= 5).sum()}/{len(d)} "
              f"({(d <= 5).mean():.0%})")

        # circular-shift permutation test: is alignment better than chance?
        n_perm = 5000
        rng = np.random.default_rng(42)
        observed_mean = d.mean()
        null_means = []
        for _ in range(n_perm):
            # shift aha positions randomly within trial range
            max_trial = max(m["n_learning_trials"] for m in metadata if m["has_aha"])
            fake_ahas = rng.integers(0, max_trial, size=len(d))
            fake_dists = []
            for fa, (r, m) in zip(fake_ahas,
                [(r, m) for r, m in zip(cohort_result["individual"], metadata)
                 if m["has_aha"]]):
                cps = r["changepoints"]["changepoints"]
                if cps:
                    fake_dists.append(min(abs(fa - cp) for cp in cps))
            if fake_dists:
                null_means.append(np.mean(fake_dists))

        if null_means:
            perm_p = np.mean([nm <= observed_mean for nm in null_means])
            print(f"  permutation p (observed <= null): {perm_p:.4f}")
    else:
        print("  no aha participants with detected changepoints")

    # ─── summary ───
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results = {
        "n_participants": len(canonical),
        "classification_summary": summary,
        "dose_response": {cond: {
            "n": len(entries),
            "n_abrupt": sum(1 for e in entries if e["label"] == "abrupt"),
            "abrupt_rate": sum(1 for e in entries if e["label"] == "abrupt") / len(entries),
        } for cond, entries in by_condition.items()},
        "persistence": persist,
        "aha_alignment": {
            "n": len(aha_cp_distances),
            "mean_distance": float(np.mean(aha_cp_distances)) if aha_cp_distances else None,
        } if aha_cp_distances else None,
    }

    for sig, desc in [
        ("S1", "transition detection"),
        ("S2", "dose-response"),
        ("S3", "persistence (controlled)"),
        ("S4", "aha-changepoint alignment"),
    ]:
        status = "ready" if canonical else "no data"
        print(f"  {sig} {desc}: {status}")

    return results


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    participants = load_participants(data_dir)
    print(f"loaded {len(participants)} participants from {data_dir}/")

    if not participants:
        print("no data found. collect data first.")
        return

    results = analyze_cohort(participants)

    if results:
        out_path = os.path.join(os.path.dirname(__file__), "results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nresults saved: {out_path}")


if __name__ == "__main__":
    main()
