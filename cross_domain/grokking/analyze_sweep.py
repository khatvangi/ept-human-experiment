#!/usr/bin/env python3
"""
analyze grokking sweep — run inference module on all 60 runs.
produces the dose-response curve and per-run classification.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference import detect_transitions
from inference.psi import compute_psi_smooth, psi_peak_stats

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def analyze_all_runs():
    summary = pd.read_csv(os.path.join(RUNS_DIR, "sweep_summary.csv"))
    print(f"loaded {len(summary)} runs")
    print(f"\ngrok rate by WD:")

    results = []

    for wd in sorted(summary["weight_decay"].unique()):
        wd_runs = summary[summary["weight_decay"] == wd]
        n_grok = wd_runs["grokked"].sum()
        print(f"  WD={wd:.2f}: {n_grok}/{len(wd_runs)} grokked")

        for _, row in wd_runs.iterrows():
            seed = int(row["seed"])
            csv_path = os.path.join(RUNS_DIR, f"run_wd{wd}_seed{seed}.csv")

            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            test_acc = df["test_acc"].values

            # subsample long runs for tractable HMM
            if len(test_acc) > 200:
                step = max(1, len(test_acc) // 200)
                test_acc_sub = test_acc[::step]
            else:
                test_acc_sub = test_acc

            r = detect_transitions(test_acc_sub)

            # also get full-res Ψ stats
            psi = compute_psi_smooth(test_acc, sigma=3.0)
            ps = psi_peak_stats(psi)

            results.append({
                "seed": seed,
                "weight_decay": wd,
                "grokked": bool(row["grokked"]),
                "final_test_acc": float(row["final_test_acc"]),
                "n_epochs": int(row["n_epochs_run"]),
                "classification": r["classification"]["label"],
                "confidence": r["classification"]["confidence"],
                "psi_z": ps["z_score"] if ps else 0,
                "psi_spike_ratio": ps.get("spike_ratio", 0) if ps else 0,
                "psi_localization": ps.get("localization", 1) if ps else 1,
                "jump_size": r["model_comparison"]["jump_size"],
                "transition_width": r["model_comparison"]["transition_width"],
                "n_changepoints": r["changepoints"]["n_changes"],
            })

    # print per-run results
    print(f"\n{'WD':>5s} {'seed':>4s} {'grok':>5s} {'class':>12s} "
          f"{'Ψ z':>6s} {'ratio':>7s} {'local':>7s} {'jump':>6s}")
    print("-" * 65)

    for r in results:
        print(f"{r['weight_decay']:5.2f} {r['seed']:4d} "
              f"{'YES' if r['grokked'] else 'no':>5s} "
              f"{r['classification']:>12s} "
              f"{r['psi_z']:6.1f} {r['psi_spike_ratio']:7.1f} "
              f"{r['psi_localization']:7.4f} {r['jump_size']:6.3f}")

    # aggregate by WD
    print(f"\n{'='*65}")
    print("DOSE-RESPONSE SUMMARY")
    print(f"{'='*65}")
    print(f"\n{'WD':>5s} {'n':>3s} {'grok%':>6s} {'abrupt':>7s} "
          f"{'gradual':>8s} {'non_lrn':>8s} {'unstable':>9s} "
          f"{'mean_Ψz':>8s} {'mean_epoch':>10s}")
    print("-" * 75)

    for wd in sorted(set(r["weight_decay"] for r in results)):
        wd_results = [r for r in results if r["weight_decay"] == wd]
        n = len(wd_results)
        n_grok = sum(1 for r in wd_results if r["grokked"])
        labels = [r["classification"] for r in wd_results]
        n_abrupt = labels.count("abrupt")
        n_gradual = labels.count("gradual")
        n_non = labels.count("non_learner")
        n_unstable = labels.count("unstable")
        mean_z = np.mean([r["psi_z"] for r in wd_results])
        mean_epoch = np.mean([r["n_epochs"] for r in wd_results])

        print(f"{wd:5.2f} {n:3d} {n_grok/n:6.0%} "
              f"{n_abrupt:>7d} {n_gradual:>8d} {n_non:>8d} {n_unstable:>9d} "
              f"{mean_z:8.1f} {mean_epoch:10.0f}")

    # save
    out_path = os.path.join(os.path.dirname(__file__), "sweep_inference_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")

    return results


if __name__ == "__main__":
    analyze_all_runs()
