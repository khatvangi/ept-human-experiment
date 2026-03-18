"""
classify.py — classify learners into transition types

uses model comparison + changepoint detection + Ψ analysis
to assign each learner to one of:

  ABRUPT:    single clear transition, changepoint model wins, Ψ spike
  GRADUAL:   smooth improvement, continuous model wins
  UNSTABLE:  multiple state switches, HMM model wins
  NON_LEARNER: no improvement, flat performance
  MIXED:     ambiguous — models disagree or evidence is weak
"""

import numpy as np
from .model_compare import fit_and_compare
from .psi import compute_psi_smooth, psi_peak_stats
from .changepoint import detect_changepoints


# learner type labels
ABRUPT = "abrupt"
GRADUAL = "gradual"
UNSTABLE = "unstable"
NON_LEARNER = "non_learner"
MIXED = "mixed"


def classify_learner(series, min_improvement=0.15, chance_level=0.5):
    """
    classify a single learner's trajectory.

    args:
        series: 1D array of performance (accuracy) over trials
        min_improvement: minimum improvement from first to last quartile
                         to count as having learned at all
        chance_level: baseline performance (e.g. 0.5 for binary classification)

    returns:
        dict with:
            label: one of ABRUPT, GRADUAL, UNSTABLE, NON_LEARNER, MIXED
            confidence: "high", "medium", "low"
            model_comparison: full output from fit_and_compare
            psi_stats: Ψ peak statistics
            changepoints: detected changepoints
            reasoning: human-readable explanation
    """
    arr = np.asarray(series, dtype=float)
    T = len(arr)

    if T < 15:
        return {
            "label": MIXED,
            "confidence": "low",
            "reasoning": f"too few trials ({T}) for reliable classification",
        }

    # step 1: check if any learning occurred
    # use first/last third (more robust than quartile with noise)
    third = max(1, T // 3)
    q_early = np.mean(arr[:third])
    q_late = np.mean(arr[-third:])
    improvement = q_late - q_early

    # also compute overall trend via linear regression
    from scipy import stats as sp_stats
    slope, _, _, slope_p, _ = sp_stats.linregress(np.arange(T), arr)
    has_trend = slope > 0.001 and slope_p < 0.05

    if improvement < min_improvement and q_late < chance_level + min_improvement and not has_trend:
        return {
            "label": NON_LEARNER,
            "confidence": "high" if improvement < 0.05 else "medium",
            "improvement": float(improvement),
            "final_performance": float(q_late),
            "reasoning": f"minimal improvement ({improvement:.3f}), "
                         f"final performance ({q_late:.3f}) near chance ({chance_level}), "
                         f"no significant trend (slope={slope:.4f}, p={slope_p:.3f})",
        }

    # step 2: model comparison
    mc = fit_and_compare(arr)

    # step 3: Ψ analysis
    psi = compute_psi_smooth(arr, sigma=3.0)
    psi_stats = psi_peak_stats(psi)

    # step 4: changepoint detection
    cp_result = detect_changepoints(arr, method="pelt")

    # step 5: classification logic
    label, confidence, reasoning = _decide(mc, psi_stats, cp_result, improvement, T)

    return {
        "label": label,
        "confidence": confidence,
        "model_comparison": mc,
        "psi_stats": psi_stats,
        "changepoints": cp_result["changepoints"],
        "improvement": float(improvement),
        "reasoning": reasoning,
    }


def _decide(mc, psi_stats, cp_result, improvement, T):
    """
    classification logic combining multiple evidence sources.

    key insight: BIC model comparison between sigmoid and step-function
    is unreliable because a steep sigmoid approximates a step. instead,
    use the sigmoid's fitted RATE parameter as the primary abruptness
    indicator, with PELT changepoints and Ψ spikes as corroboration.

    evidence sources (in priority order):
      1. sigmoid transition width (from continuous model fit)
      2. PELT changepoint detection
      3. Ψ spike z-score
      4. jump magnitude at changepoint
      5. HMM state switching pattern
    """
    n_cp = cp_result["n_changes"]

    # --- extract evidence signals ---

    # sigmoid steepness
    trans_width = mc.get("transition_width", T)
    is_sharp_sigmoid = trans_width < 0.12 * T  # strict: < 12% of total

    # PELT: exactly 1 changepoint = single transition; 2+ = complex
    has_single_cp = n_cp == 1
    has_multi_cp = n_cp >= 2

    # Ψ spike (z > 3 for high confidence, > 2 for medium)
    psi_z = psi_stats.get("z_score", 0) if psi_stats else 0
    psi_loc = psi_stats.get("localization", 1.0) if psi_stats else 1.0
    has_strong_spike = psi_z > 3.0 and psi_loc < 0.25
    has_weak_spike = psi_z > 2.0 and psi_loc < 0.3

    # jump size (strict threshold: must be > 0.25 for abrupt)
    jump = mc["jump_size"]
    has_large_jump = jump > 0.25
    has_small_jump = 0.15 < jump <= 0.25

    # HMM wins?
    hmm_wins = mc["best_model"] == "hmm"

    # segment monotonicity: do PELT segments go monotonically up?
    seg_means = cp_result["segment_means"]
    is_monotone = all(seg_means[i] <= seg_means[i + 1] + 0.05
                      for i in range(len(seg_means) - 1))

    # --- classification rules (order matters) ---

    # OVERRIDE: if segments are monotone AND there's a massive Ψ spike,
    # this is abrupt even if HMM wins (grokking has 3 monotone regimes).
    # HMM wins because it captures the plateau-transition-plateau structure
    # better than a single step, but the transition is still abrupt.
    if is_monotone and has_strong_spike and has_large_jump and is_sharp_sigmoid:
        return ABRUPT, "high", (
            f"monotone trajectory with sharp transition: "
            f"width={trans_width:.0f}, Ψ z={psi_z:.1f}, jump={jump:.3f}, "
            f"segments={[f'{m:.2f}' for m in seg_means]}"
        )

    # UNSTABLE: HMM wins OR multiple changepoints with non-monotone segments
    if hmm_wins and has_multi_cp and not is_monotone:
        return UNSTABLE, "high", (
            f"HMM preferred + {n_cp} non-monotone changepoints, state-switching"
        )
    if hmm_wins and not is_monotone:
        return UNSTABLE, "medium", (
            f"HMM preferred + non-monotone trajectory"
        )
    if has_multi_cp and not is_monotone:
        return UNSTABLE, "medium", (
            f"{n_cp} changepoints with non-monotone segments: "
            f"means={[f'{m:.2f}' for m in seg_means]}"
        )

    # ABRUPT: sharp sigmoid + single changepoint + large jump
    if is_sharp_sigmoid and has_single_cp and has_large_jump:
        return ABRUPT, "high", (
            f"sharp transition: width={trans_width:.0f}, "
            f"1 changepoint at {cp_result['changepoints'][0]}, "
            f"jump={jump:.3f}, Ψ z={psi_z:.1f}"
        )
    # abrupt with slightly weaker evidence
    if is_sharp_sigmoid and has_large_jump and has_strong_spike:
        return ABRUPT, "high", (
            f"sharp sigmoid + large jump + Ψ spike: "
            f"width={trans_width:.0f}, jump={jump:.3f}, z={psi_z:.1f}"
        )
    if has_single_cp and has_large_jump and has_strong_spike:
        return ABRUPT, "medium", (
            f"single changepoint + large jump + spike: "
            f"jump={jump:.3f}, z={psi_z:.1f}"
        )

    # GRADUAL: wide transition + monotone + learning occurred
    if trans_width > 0.25 * T and is_monotone and improvement > 0.10:
        return GRADUAL, "high", (
            f"wide transition: width={trans_width:.0f} trials, "
            f"monotone segments, improvement={improvement:.3f}"
        )
    if not is_sharp_sigmoid and improvement > 0.10 and not has_large_jump:
        return GRADUAL, "medium", (
            f"non-sharp sigmoid: width={trans_width:.0f}, "
            f"jump={jump:.3f} (below threshold), improvement={improvement:.3f}"
        )

    # UNSTABLE: remaining multi-cp or HMM cases
    if hmm_wins:
        return UNSTABLE, "low", f"HMM preferred but pattern unclear"
    if has_multi_cp:
        return UNSTABLE, "low", (
            f"{n_cp} changepoints, means={[f'{m:.2f}' for m in seg_means]}"
        )

    # MIXED: ambiguous
    return MIXED, "low", (
        f"ambiguous: width={trans_width:.0f}, jump={jump:.3f}, "
        f"n_cp={n_cp}, psi_z={psi_z:.1f}"
    )


def classify_cohort(trajectories, **kwargs):
    """
    classify a list of learners and return summary statistics.

    args:
        trajectories: list of 1D arrays (one per participant)
        **kwargs: passed to classify_learner

    returns:
        dict with:
            classifications: list of per-participant classification dicts
            summary: counts and proportions for each type
            abrupt_fraction: fraction classified as ABRUPT
    """
    classifications = [classify_learner(t, **kwargs) for t in trajectories]

    counts = {}
    for c in classifications:
        lbl = c["label"]
        counts[lbl] = counts.get(lbl, 0) + 1

    n = len(classifications)
    summary = {lbl: {"count": cnt, "fraction": cnt / n}
               for lbl, cnt in counts.items()}

    return {
        "classifications": classifications,
        "summary": summary,
        "n_total": n,
        "abrupt_fraction": counts.get(ABRUPT, 0) / n if n > 0 else 0,
    }
