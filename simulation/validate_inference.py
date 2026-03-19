#!/usr/bin/env python3
"""
validate_inference.py — prove the inference module recovers known learner types

generates synthetic learners with known ground truth, runs the full
inference pipeline, and reports classification accuracy.

this is the "it's not circular" proof. if the inference module can't
recover what we put in, it can't be trusted on real data.

usage:
    python simulation/validate_inference.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulation.synthetic_learners import generate_cohort
from inference.pipeline import detect_transitions


def validate(n_per_type=30, n_trials=100, verbose=True):
    """
    run full validation:
      1. generate synthetic cohort with known labels
      2. run inference pipeline on each
      3. compute confusion matrix
      4. report recovery accuracy
    """
    print("=" * 70)
    print("INFERENCE VALIDATION: synthetic learner recovery")
    print("=" * 70)

    cohort = generate_cohort(n_per_type=n_per_type, n_trials=n_trials, seed=42)
    print(f"\ngenerated {len(cohort)} synthetic learners "
          f"({n_per_type} x 5 types, {n_trials} trials each)")

    # mapping from ground truth to expected inference labels
    # note: false_aha should be classified as gradual (its accuracy IS gradual)
    # the false_aha distinction only appears in cross-channel divergence
    gt_to_expected = {
        "gradual": "gradual",
        "abrupt": "abrupt",
        "non_learner": "non_learner",
        "unstable": "unstable",
        "false_aha": "gradual",  # accuracy is gradual; aha is metacognitive
        "power_law": "gradual",  # adversarial: out-of-model-family, but still gradual
        "double_sigmoid": "gradual",  # adversarial: two moderate steps, not sharp enough for abrupt
    }

    # run inference
    results = []
    for p in cohort:
        r = detect_transitions(
            accuracy_series=p["accuracy"],
            confidence_series=p["confidence"],
            rt_series=p["rt"],
            transfer_series=p["transfer"],
        )
        results.append({
            "id": p["participant_id"],
            "ground_truth": p["ground_truth"],
            "expected": gt_to_expected[p["ground_truth"]],
            "predicted": r["classification"]["label"],
            "confidence": r["classification"]["confidence"],
            "model": r["model_comparison"]["best_model"],
            "delta_bic": r["model_comparison"]["delta_bic"],
            "convergence": r["convergence"]["convergence_score"] if r["convergence"] else None,
        })

    # confusion matrix
    all_labels = sorted(set(r["expected"] for r in results) |
                        set(r["predicted"] for r in results))

    confusion = {}
    for gt in all_labels:
        confusion[gt] = {}
        for pred in all_labels:
            confusion[gt][pred] = 0

    for r in results:
        gt = r["expected"]
        pred = r["predicted"]
        if gt in confusion and pred in confusion[gt]:
            confusion[gt][pred] += 1

    # print confusion matrix
    print(f"\n{'':15s}", end="")
    for pred in all_labels:
        print(f"{pred:>12s}", end="")
    print(f"{'accuracy':>12s}")
    print("-" * (15 + 12 * (len(all_labels) + 1)))

    total_correct = 0
    total = 0
    per_class_acc = {}

    for gt in all_labels:
        print(f"{gt:15s}", end="")
        row_total = sum(confusion[gt].values())
        row_correct = confusion[gt].get(gt, 0)

        for pred in all_labels:
            count = confusion[gt].get(pred, 0)
            print(f"{count:12d}", end="")

        acc = row_correct / row_total if row_total > 0 else 0
        per_class_acc[gt] = acc
        print(f"{acc:12.1%}")
        total_correct += row_correct
        total += row_total

    overall_acc = total_correct / total if total > 0 else 0
    print("-" * (15 + 12 * (len(all_labels) + 1)))
    print(f"{'overall':15s}", end="")
    print(f"{'':>{12 * len(all_labels)}s}", end="")
    print(f"{overall_acc:12.1%}")

    # false_aha specific analysis: do false_aha have divergent channels?
    print("\n" + "=" * 70)
    print("FALSE AHA DETECTION (cross-channel divergence)")
    print("=" * 70)

    false_aha_results = [r for r in results if r["ground_truth"] == "false_aha"]
    true_abrupt_results = [r for r in results if r["ground_truth"] == "abrupt"]

    if false_aha_results:
        fa_conv = [r["convergence"] for r in false_aha_results if r["convergence"] is not None]
        ta_conv = [r["convergence"] for r in true_abrupt_results if r["convergence"] is not None]

        if fa_conv and ta_conv:
            print(f"  false_aha mean convergence: {np.mean(fa_conv):.3f}")
            print(f"  true abrupt mean convergence: {np.mean(ta_conv):.3f}")
            print(f"  (lower convergence in false_aha = channels disagree = detectable)")
        else:
            print("  insufficient convergence data for comparison")

    # summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  overall accuracy: {overall_acc:.1%}")
    for gt, acc in per_class_acc.items():
        status = "PASS" if acc >= 0.60 else "MARGINAL" if acc >= 0.40 else "FAIL"
        print(f"  {gt:15s}: {acc:.1%} [{status}]")

    threshold = 0.60
    passed = overall_acc >= threshold
    print(f"\n  {'VALIDATION PASSED' if passed else 'VALIDATION FAILED'} "
          f"(threshold: {threshold:.0%})")

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "confusion": confusion,
        "passed": passed,
        "n_total": total,
        "individual": results,
    }


if __name__ == "__main__":
    validate(n_per_type=30, n_trials=100, verbose=True)
