#!/usr/bin/env python3
"""
analyze grokking sweep — run inference module on all 60 runs.

classifies each run and produces the dose-response table:
WD value → fraction abrupt / gradual / non-learner.
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

import sys
import os
import glob
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference import detect_transitions
from inference.psi import compute_psi_smooth, psi_peak_stats


def analyze_all_runs(runs_dir):
    results = []

    csv_files = sorted(glob.glob(os.path.join(runs_dir, "run_wd*.csv")))
    print(f"found {len(csv_files)} run files\n")

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        parts = fname.replace("run_wd", "").replace(".csv", "").split("_seed")
        wd = float(parts[0])
        seed = int(parts[1])

        df = pd.read_csv(csv_path)
        test_acc = df["test_acc"].values

        # subsample long runs for tractable HMM
        if len(test_acc) > 200:
            step = max(1, len(test_acc) // 200)
            test_acc_sub = test_acc[::step]
        else:
            test_acc_sub = test_acc

        result = detect_transitions(test_acc_sub)

        # full-resolution Ψ
        psi_full = compute_psi_smooth(test_acc, sigma=3.0)
        ps_full = psi_peak_stats(psi_full)

        label = result["classification"]["label"]
        conf = result["classification"]["confidence"]
        grokked = df["test_acc"].iloc[-1] > 0.95

        results.append({
            "wd": wd, "seed": seed, "label": label, "confidence": conf,
            "grokked": grokked, "final_test_acc": float(df["test_acc"].iloc[-1]),
            "n_epochs": int(df["epoch"].iloc[-1]),
            "psi_z": ps_full["z_score"] if ps_full else 0,
            "psi_spike_ratio": ps_full["spike_ratio"] if ps_full else 0,
            "psi_localization": ps_full["localization"] if ps_full else 1.0,
            "jump_size": result["model_comparison"]["jump_size"],
            "transition_width": result["model_comparison"]["transition_width"],
        })

    return results


def print_results(results):
    # dose-response summary
    print("=" * 70)
    print("DOSE-RESPONSE: transition topology by weight decay")
    print("=" * 70)

    wd_values = sorted(set(r["wd"] for r in results))
    print(f"\n{'WD':>6s} | {'n':>3s} | {'abrupt':>8s} | {'gradual':>8s} | "
          f"{'non_lrn':>8s} | {'other':>8s} | {'grokked':>8s}")
    print("-" * 70)

    for wd in wd_values:
        runs = [r for r in results if r["wd"] == wd]
        n = len(runs)
        counts = {}
        for r in runs:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        n_grok = sum(1 for r in runs if r["grokked"])
        n_other = n - counts.get("abrupt", 0) - counts.get("gradual", 0) - counts.get("non_learner", 0)

        print(f"{wd:6.2f} | {n:3d} | "
              f"{counts.get('abrupt', 0):3d}({counts.get('abrupt', 0)/n:3.0%})  | "
              f"{counts.get('gradual', 0):3d}({counts.get('gradual', 0)/n:3.0%})  | "
              f"{counts.get('non_learner', 0):3d}({counts.get('non_learner', 0)/n:3.0%})  | "
              f"{n_other:3d}({n_other/n:3.0%})  | "
              f"{n_grok:3d}({n_grok/n:3.0%})")

    # Ψ stats for grokked runs
    grokked = [r for r in results if r["grokked"]]
    if grokked:
        zs = [r["psi_z"] for r in grokked]
        ratios = [r["psi_spike_ratio"] for r in grokked]
        locs = [r["psi_localization"] for r in grokked]
        print(f"\nΨ statistics (grokked runs, n={len(grokked)}):")
        print(f"  z-score:      {np.mean(zs):.1f} ± {np.std(zs):.1f}")
        print(f"  spike ratio:  {np.mean(ratios):.1f} ± {np.std(ratios):.1f}")
        print(f"  localization: {np.mean(locs):.4f} ± {np.std(locs):.4f}")

    # per-run detail
    print(f"\n{'WD':>6s} {'seed':>4s} {'label':>12s} {'grok':>5s} "
          f"{'Ψ z':>6s} {'ratio':>7s} {'jump':>6s}")
    print("-" * 52)
    for r in sorted(results, key=lambda x: (x["wd"], x["seed"])):
        print(f"{r['wd']:6.2f} {r['seed']:4d} {r['label']:>12s} "
              f"{'Y' if r['grokked'] else 'N':>5s} "
              f"{r['psi_z']:6.1f} {r['psi_spike_ratio']:7.1f} {r['jump_size']:6.3f}")


if __name__ == "__main__":
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    results = analyze_all_runs(runs_dir)
    print_results(results)

    out_path = os.path.join(os.path.dirname(__file__), "sweep_inference_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")
