#!/usr/bin/env python3
"""
cross-domain transition analysis — apply the same inference module
to grokking (neural network), mouse maze (rodent), and any future domain.

the core claim: the same statistical machinery detects transition
signatures across fundamentally different learning systems.
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import detect_transitions
from inference.psi import compute_psi_smooth, psi_peak_stats


def analyze_grokking():
    """
    analyze grokking test accuracy trajectory.
    expected: abrupt transition with massive Ψ spike.
    """
    print("=" * 70)
    print("DOMAIN 1: NEURAL NETWORK GROKKING (mod-97 addition)")
    print("=" * 70)

    csv_path = "/storage/EPT/grokking_timeseries.csv"
    df = pd.read_csv(csv_path)

    # use test accuracy as the learning curve
    test_acc = df["test_acc"].values

    # subsample to ~200 points for tractable HMM
    if len(test_acc) > 200:
        step = len(test_acc) // 200
        test_acc_sub = test_acc[::step]
        epoch_sub = df["epoch"].values[::step]
    else:
        test_acc_sub = test_acc
        epoch_sub = df["epoch"].values

    result = detect_transitions(test_acc_sub)

    print(f"\n  data: {len(test_acc)} epochs (subsampled to {len(test_acc_sub)})")
    print(f"  classification: {result['classification']['label']} "
          f"({result['classification']['confidence']})")
    print(f"  reasoning: {result['classification']['reasoning']}")

    if result["psi"]:
        ps = result["psi"]
        print(f"\n  Ψ peak:")
        print(f"    z-score: {ps.get('z_score', 'N/A'):.1f}")
        print(f"    spike ratio: {ps.get('spike_ratio', 'N/A'):.1f}")
        print(f"    localization: {ps.get('localization', 'N/A'):.3f}")
        print(f"    peak at sample: {ps.get('peak_idx', 'N/A')}")

    mc = result["model_comparison"]
    print(f"\n  model comparison:")
    print(f"    best: {mc['best_model']} ({mc['evidence_strength']})")
    print(f"    ΔBIC(cont-cp): {mc['delta_bic']:.1f}")
    print(f"    transition width: {mc['transition_width']:.0f} samples")
    print(f"    jump size: {mc['jump_size']:.3f}")

    cp = result["changepoints"]
    print(f"\n  changepoints: {cp['changepoints']} "
          f"(segment means: {[f'{m:.3f}' for m in cp['segment_means']]})")

    # also compute Ψ on full resolution
    psi_full = compute_psi_smooth(test_acc, sigma=5.0)
    ps_full = psi_peak_stats(psi_full)
    if ps_full:
        grok_epoch = df["epoch"].values[ps_full["peak_idx"]]
        print(f"\n  full-resolution Ψ:")
        print(f"    peak at epoch {grok_epoch}")
        print(f"    z-score: {ps_full['z_score']:.1f}")
        print(f"    spike ratio: {ps_full['spike_ratio']:.1f}")
        print(f"    localization: {ps_full['localization']:.4f}")

    return {
        "domain": "grokking",
        "system": "neural_network",
        "n_datapoints": len(test_acc),
        "classification": result["classification"]["label"],
        "psi_z": ps_full["z_score"] if ps_full else None,
        "psi_spike_ratio": ps_full["spike_ratio"] if ps_full else None,
        "psi_localization": ps_full["localization"] if ps_full else None,
        "transition_width_frac": mc["transition_width"] / len(test_acc_sub),
        "jump_size": mc["jump_size"],
        "n_changepoints": cp["n_changes"],
    }


def analyze_mouse_maze():
    """
    analyze per-mouse learning curves from Rosenberg et al. labyrinth.
    expected: some mice show abrupt "insight," others gradual.
    """
    print("\n" + "=" * 70)
    print("DOMAIN 2: MOUSE MAZE LEARNING (Rosenberg et al. 2021)")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__),
                            "mouse_maze/mouse_trajectories.csv")
    df = pd.read_csv(csv_path)

    print(f"\n  data: {len(df)} bouts across {df['mouse_id'].nunique()} mice")
    print(f"  groups: {df['group'].value_counts().to_dict()}")

    # analyze per-mouse learning curves
    # key metric: sliding-window probability of making a home run
    # (this is what the Rosenberg paper shows as the learning curve)
    results_per_mouse = []
    from scipy.ndimage import gaussian_filter1d

    for mouse_id in sorted(df["mouse_id"].unique()):
        mouse_df = df[df["mouse_id"] == mouse_id].copy()

        # binary: did a home run occur in this bout? (distance > 0 = yes)
        has_hr = (mouse_df["home_run_distance"].fillna(0) > 0).astype(float).values

        if len(has_hr) < 20:
            continue

        # sliding window home-run rate (window=15 bouts)
        window = min(15, len(has_hr) // 4)
        kernel = np.ones(window) / window
        hr_rate = np.convolve(has_hr, kernel, mode="same")

        # smooth a bit more for stability
        hrd_smooth = gaussian_filter1d(hr_rate, sigma=2)

        result = detect_transitions(hrd_smooth)

        label = result["classification"]["label"]
        conf = result["classification"]["confidence"]
        psi_z = result["psi"]["z_score"] if result["psi"] else 0
        jump = result["model_comparison"]["jump_size"]
        group = mouse_df["group"].iloc[0]

        results_per_mouse.append({
            "mouse_id": mouse_id,
            "group": group,
            "n_bouts": len(has_hr),
            "label": label,
            "confidence": conf,
            "psi_z": psi_z,
            "jump_size": jump,
            "mean_performance": float(hr_rate.mean()),
            "final_performance": float(hr_rate[-20:].mean()),
        })

        status = f"{label:12s} (z={psi_z:.1f}, jump={jump:.2f})"
        print(f"  {mouse_id:4s} ({group:10s}): {len(has_hr):3d} bouts → {status}")

    # summary
    labels = [r["label"] for r in results_per_mouse]
    label_counts = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\n  classification summary:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl}: {cnt}/{len(results_per_mouse)} "
              f"({cnt/len(results_per_mouse):.0%})")

    # compare rewarded vs unrewarded
    for group in ["rewarded", "unrewarded"]:
        grp = [r for r in results_per_mouse if r["group"] == group]
        n_abrupt = sum(1 for r in grp if r["label"] == "abrupt")
        print(f"  {group}: {n_abrupt}/{len(grp)} abrupt")

    return {
        "domain": "mouse_maze",
        "system": "rodent",
        "n_subjects": len(results_per_mouse),
        "per_mouse": results_per_mouse,
        "classification_counts": label_counts,
    }


def cross_domain_summary(results):
    """
    print the cross-domain comparison table.
    """
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON")
    print("=" * 70)

    print(f"\n  {'domain':<20s} {'system':<15s} {'classification':<15s} "
          f"{'Ψ z-score':<12s} {'Ψ ratio':<12s} {'localization':<12s}")
    print("  " + "-" * 86)

    for r in results:
        if "psi_z" in r and r["psi_z"] is not None:
            print(f"  {r['domain']:<20s} {r['system']:<15s} "
                  f"{r.get('classification', 'N/A'):<15s} "
                  f"{r['psi_z']:<12.1f} "
                  f"{r.get('psi_spike_ratio', 0):<12.1f} "
                  f"{r.get('psi_localization', 0):<12.4f}")
        elif "per_mouse" in r:
            # summarize mouse data
            abrupt_mice = [m for m in r["per_mouse"] if m["label"] == "abrupt"]
            if abrupt_mice:
                avg_z = np.mean([m["psi_z"] for m in abrupt_mice])
                print(f"  {r['domain']:<20s} {r['system']:<15s} "
                      f"{'mixed':<15s} "
                      f"{avg_z:<12.1f} "
                      f"{'N/A':<12s} "
                      f"{'N/A':<12s}")
                print(f"  {'':20s} {'':15s} "
                      f"({len(abrupt_mice)}/{r['n_subjects']} abrupt)")

    print("\n  key: same inference code, different domains, same signature types.")


if __name__ == "__main__":
    all_results = []

    grok = analyze_grokking()
    all_results.append(grok)

    maze = analyze_mouse_maze()
    all_results.append(maze)

    cross_domain_summary(all_results)

    # save
    out_path = os.path.join(os.path.dirname(__file__), "cross_domain_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  results saved: {out_path}")
