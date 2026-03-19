# learning-transition-inference

A domain-agnostic framework for inferring transition topologies in learning trajectories.

The same inference engine classifies learning dynamics as abrupt, gradual, unstable, or non-learner — whether the learner is a neural network, a mouse, or a human. The framework does not assume all learning has one shape. It measures what shape each trajectory actually has, then asks whether different systems exhibit different topologies.

## Core claim

We present a unified inference pipeline for detecting transition signatures in learning curves. Applied to two empirical domains (neural network grokking and rodent maze learning) and validated on synthetic data, the framework identifies distinct transition topologies: abrupt reorganization in grokking (dose-response confirmed across 60 runs), and state-switching dynamics in mouse maze learning. A human Artificial Grammar Learning experiment is deployed and awaiting data collection.

## Quick start

```bash
pip install -r requirements.txt

# validate the inference module on synthetic learners (5 types, 76% recovery)
python simulation/validate_inference.py

# run cross-domain analysis (grokking + mouse maze)
python cross_domain/analyze_all.py

# analyze grokking dose-response sweep (60 runs, 6 weight decay values)
python cross_domain/grokking/analyze_sweep.py

# run human AGL analysis (once participant data is collected)
python tasks/agl/analysis/analyze_agl.py /path/to/data/
```

## Repository structure

```
inference/                     THE CORE — domain-agnostic transition detection
  __init__.py                  public API: from inference import detect_transitions
  psi.py                       Ψ(t) velocity order parameter: |d(performance)/dt|
  changepoint.py               PELT (ruptures) + Bayesian online changepoint detection
  model_compare.py             fit sigmoid vs step-function vs 2-state HMM, select by BIC
  classify.py                  combine evidence → abrupt / gradual / unstable / non-learner
  persistence.py               post-transition stability, with confound control
  convergence.py               cross-channel alignment with permutation test
  pipeline.py                  unified detect_transitions() and detect_transitions_cohort()

cross_domain/                  empirical results
  grokking/
    train_grokking.py          training script (mod-97 addition, small transformer)
    analyze_sweep.py           run inference on all 60 runs, produce dose-response table
    runs/                      60 per-run CSVs (epoch, train_acc, test_acc, loss)
    sweep_summary.csv          one row per run: seed, WD, grokked, final_test_acc
  mouse_maze/
    extract_trajectories.py    extract home-run learning curves from Rosenberg et al.
    mouse_trajectories.csv     3520 bouts across 19 mice (rewarded + unrewarded)
  analyze_all.py               run inference on both domains, print comparison table

simulation/                    synthetic validation
  synthetic_learners.py        generate 5 learner types with known ground truth
  validate_inference.py        recovery test: 76% overall (abrupt 95%, non-learner 100%)

tasks/agl/                     human AGL experiment
  web/
    index.html                 complete experiment UI (consent → exposure → test → debrief)
    grammars.js                3 FSA grammars (easy/medium/hard), string generation
    experiment.js              trial engine with exposure phase, confidence probes, aha button
  analysis/
    analyze_agl.py             thin adapter: load participant JSON → inference pipeline

docs/
  EXPERIMENT_DESIGN.md         AGL protocol, transition-signature predictions (S1–S4)
  LITERATURE_REVIEW.md         cross-domain literature (AGL, insight, grokking, animal learning)

ept_theory_docs/               EPT theoretical framework (propositions, kinetics, evidence)
recruitment/                   consent form (Common Rule), flyer, Prolific description
scispace_data/                 literature search (230 files, bibliographic metadata)
```

## Inference module

### API

```python
from inference import detect_transitions

result = detect_transitions(
    accuracy_series,              # required: 1D array of performance over trials
    confidence_series=None,       # optional: metacognitive ratings
    rt_series=None,               # optional: reaction times (ms)
    transfer_series=None,         # optional: post-perturbation test data
)
```

### Output

```python
result["classification"]["label"]       # "abrupt", "gradual", "non_learner", "unstable", "mixed"
result["classification"]["confidence"]  # "high", "medium", "low"
result["classification"]["reasoning"]   # human-readable explanation

result["psi"]                           # Ψ peak stats: z_score, spike_ratio, localization
result["changepoints"]                  # PELT results: indices, segment means
result["model_comparison"]              # BIC for sigmoid vs step vs HMM, transition width
result["persistence"]                   # transfer test with confound control (if transfer data given)
result["convergence"]                   # cross-channel alignment with permutation p-value
```

### What it does

1. **Ψ (velocity order parameter)**: Gaussian-smoothed |d(performance)/dt|. Characterizes the spike: z-score, spike ratio (peak/baseline), localization (fraction of trajectory in spike).

2. **Changepoint detection**: PELT with conservative auto-penalty (BIC-scaled, elbow method). Finds major mean-shifts without over-segmenting noisy data. Also supports Bayesian Online Changepoint Detection.

3. **Model comparison**: Fits three models to each trajectory:
   - Sigmoid (gradual): 4 params — base, ceiling, rate, midpoint
   - Piecewise constant (abrupt): 3 params — mu_before, mu_after, changepoint
   - 2-state Gaussian HMM (unstable): 6 params — 2 means, 2 variances, 2 transition probs

   Selects by BIC. Extracts sigmoid steepness (transition width) as a primary abruptness indicator, since BIC comparison alone is unreliable (steep sigmoid approximates a step function).

4. **Classification**: Multi-evidence decision combining sigmoid width, PELT changepoint count, Ψ spike z-score, jump magnitude, and segment monotonicity. Does not rely on any single indicator.

5. **Persistence**: Tests whether post-transition performance holds after perturbation (feedback removal, novel stimuli). Includes a controlled version that regresses out learning performance before testing transition status — addressing the confound that "better learners both transition and transfer better."

6. **Cross-channel convergence**: Detects changepoints independently in accuracy, confidence, and RT, then tests whether they align temporally. Permutation test (5000 iterations) assesses whether alignment exceeds chance.

### Validation

Synthetic learner recovery (n=100, 20 per type, 100 trials each):

| Learner type | Recovery rate | Notes |
|---|---|---|
| Abrupt | 95% | Sharp sigmoid + changepoint + Ψ spike |
| Gradual | 80% | Wide sigmoid, no significant jump |
| Non-learner | 100% | Flat trajectory near chance |
| Unstable | 25% | Inherently ambiguous at low switch rates |
| **Overall** | **76%** | |

False-aha detection: synthetic "false aha" learners (gradual accuracy, sudden confidence jump) show convergence score 0.38 vs 0.58 for true abrupt — channels disagree when the transition is metacognitive rather than behavioral.

**Known limitation**: Unstable classification (25%) is weak because stochastic switching with low probability (3% per trial) often produces trajectories with 0–1 switches, indistinguishable from abrupt or non-learner. This is an inherent ambiguity, not a detection failure.

## Cross-domain results

### Summary

| Domain | System | N | Topology | Key statistic |
|---|---|---|---|---|
| Grokking | neural network | 60 runs | abrupt (dose-response) | Ψ z = 14.3 ± 11.4 |
| Mouse maze | rodent | 19 mice | unstable (state-switching) | Ψ z = 3–7 |
| Human AGL | human | pending | ? | experiment deployed |

The framework identifies different topologies in different systems. This is a feature: it means the method discriminates rather than forcing all dynamics into one bucket.

### Grokking dose-response (60 runs)

Small transformer on modular addition (mod 97). 10 seeds per weight decay value, 7500 max epochs, trained on 2x Titan RTX.

| Weight decay | Grokked | Abrupt rate | Notes |
|---|---|---|---|
| 0.00 | 1/10 (10%) | 0% | Almost no generalization |
| 0.01 | 2/10 (20%) | 0% | Rare grokking, classified gradual |
| 0.03 | 10/10 (100%) | 30% | Threshold: all grok, transition sharpness varies |
| 0.10 | 10/10 (100%) | 40% | Moderate abruptness |
| 0.30 | 10/10 (100%) | 0% | Groks so fast it looks gradual when subsampled |
| 1.00 | 10/10 (100%) | 60% | Strongest WD, sharpest transitions |

Ψ statistics (43 grokked runs): z = 14.3 ± 11.4, spike ratio = 24.3 ± 22.7.

The WD = 0.30 result is informative: all 10 runs grok perfectly, but 90% are classified "gradual" because the transition happens so early (~50–100 epochs) that it occupies a large fraction of the subsampled trajectory. The classifier correctly identifies this as smooth improvement at that resolution. This validates that the framework measures transition topology, not just whether learning occurred.

### Mouse maze (Rosenberg et al. 2021)

19 mice in a complex labyrinth, data from [Rosenberg et al. 2021](https://doi.org/10.7554/eLife.66175). Home-run rate (sliding window) used as learning metric.

- 18/19 classified as **unstable** (state-switching)
- 1/19 classified as non-learner
- 0/19 classified as abrupt

The original paper describes "sudden insight" in mice, but the bout-level trajectory data shows intermittent exploratory efficiency — the framework correctly identifies this as state-switching rather than a single clean transition.

## Human experiment (AGL)

### Design

Classic Reber Artificial Grammar Learning with transition-signature diagnostics.

**Phases**: consent → instructions → **exposure** (25 grammatical strings, 3 sec each) → practice (10 trials with feedback) → learning (100 trials with feedback) → transfer (40 novel strings, no feedback) → old-string retest (20 trials, no feedback) → debrief

**Between-subjects**: 3 grammar difficulty conditions (easy/medium/hard, 3/5/7-state FSA).

**Measurements**: trial-level accuracy, RT (performance.now()), confidence every 10 trials, self-reported aha button, debrief survey.

**Target N**: 60–90 participants (20–30 per condition), 25–30 minutes each.

### Pre-registered analyses

All analyses use `inference/`. The AGL adapter (`tasks/agl/analysis/analyze_agl.py`) converts participant JSON to the canonical format and calls the pipeline.

| Test | Question | Method |
|---|---|---|
| S1: Transition detection | What topology does each participant's learning exhibit? | `detect_transitions()` per participant |
| S2: Dose-response | Does grammar difficulty affect transition rates? | Chi-squared on topology × condition |
| S3: Persistence | Do abrupt learners transfer better, controlling for learning ability? | Residual transfer after regressing out final accuracy |
| S4: Aha-changepoint alignment | Does self-reported aha coincide with detected changepoints? | Circular-shift permutation on |aha – nearest CP| |

Bonferroni correction: α = 0.0125 per test.

### What each outcome would mean

- All gradual → human AGL is continuous; transition signatures are not universal
- Some abrupt, some gradual → topology varies across learners (like mouse maze)
- Abrupt with persistence → framework applies to humans
- Abrupt without persistence → abruptness without stability (different from grokking)
- Aha misaligned with changepoints → self-report is unreliable as transition marker

### Deployment

Live at `thebeakers.com/study/`. Data auto-submits to backend API. Condition assigned via URL parameter (`?condition=easy`). IRB consent form follows USF Common Rule template (45 CFR 46).

## Terminology

This project deliberately separates measurement from interpretation:

| Term | Meaning in this framework |
|---|---|
| Transition signature | Measurable behavioral pattern (Ψ spike, changepoint, persistence) |
| Transition topology | Classification of a learning trajectory (abrupt, gradual, unstable) |
| Dose-response | Parameter sensitivity: transition rate varies with task difficulty or regularization |
| Persistence | Post-transition performance holds after perturbation, controlling for ability |
| Convergence | Multiple measurement channels show changepoints at the same time |

We avoid "phase transition," "hysteresis," and "universality" in their strict physical senses. The EPT theoretical framework (in `ept_theory_docs/`) provides one interpretation of these signatures; it is not the measurement layer.

## Dependencies

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
ruptures>=1.1.8
hmmlearn>=0.3
```

Optional for cross-domain analysis: `pandas`.

## Citation

If you use the inference module or cross-domain data, please cite:

```
@software{learning_transition_inference,
  title={learning-transition-inference: Domain-agnostic detection of transition topologies in learning trajectories},
  url={https://github.com/khatvangi/learning-transition-inference},
  year={2026}
}
```

## License

MIT
