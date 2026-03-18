# learning-transition-inference

a domain-agnostic framework for detecting transition signatures in learning
trajectories, applied across neural networks (grokking), rodent maze learning,
and human cognition (AGL experiment).

`inference/` is the reusable core. `tasks/agl/` is the current human-data adapter.
`cross_domain/` contains empirical results from grokking (60 runs) and mouse maze (19 mice).

## quick start

```bash
pip install -r requirements.txt

# validate the inference module on synthetic learners
python simulation/validate_inference.py

# run cross-domain analysis (grokking + mouse maze)
python cross_domain/analyze_all.py

# analyze grokking dose-response sweep (60 runs)
python cross_domain/grokking/analyze_sweep.py

# run the AGL experiment analysis (once human data is collected)
python tasks/agl/analysis/analyze_agl.py data/
```

## directory structure

```
inference/                  — THE CORE: domain-agnostic transition detection
  psi.py                    — Ψ(t) velocity order parameter
  changepoint.py            — PELT + Bayesian online changepoint detection
  model_compare.py          — continuous vs changepoint vs HMM (BIC comparison)
  classify.py               — learner type classification (abrupt/gradual/unstable/non-learner)
  persistence.py            — post-transition stability tests
  convergence.py            — cross-channel alignment with permutation test
  pipeline.py               — unified detect_transitions() interface

cross_domain/               — empirical results across domains
  grokking/                 — 60 runs (10 seeds × 6 WD), dose-response confirmed
  mouse_maze/               — 19 mice from Rosenberg et al. 2021

simulation/                 — synthetic validation
  synthetic_learners.py     — 5 learner types with known ground truth
  validate_inference.py     — recovery test (76% overall, 95% abrupt)

tasks/agl/                  — human AGL experiment
  web/                      — complete web experiment (index.html, grammars.js, experiment.js)
  analysis/analyze_agl.py   — thin adapter: loads AGL JSON → inference pipeline

docs/                       — experiment design and literature
  EXPERIMENT_DESIGN.md      — AGL protocol with transition-signature predictions
  LITERATURE_REVIEW.md      — cross-domain literature review

ept_theory_docs/            — EPT theoretical framework
recruitment/                — consent form, flyer, Prolific description
scispace_data/              — literature search data (230 files, bibliographic)
```

## inference module

the inference module detects transition signatures in any 1D learning curve:

```python
from inference import detect_transitions

result = detect_transitions(
    accuracy_series=acc,          # required: performance over trials
    confidence_series=conf,       # optional: metacognitive ratings
    rt_series=rt,                 # optional: reaction times
    transfer_series=transfer,     # optional: post-learning test data
)

print(result["classification"]["label"])  # "abrupt", "gradual", "non_learner", etc.
print(result["model_comparison"])         # continuous vs changepoint vs HMM
print(result["persistence"])              # did the transition hold?
print(result["convergence"])              # do channels agree on timing?
```

### what it does

1. **Ψ (velocity)**: computes |d(performance)/dt| — the instantaneous learning rate
2. **changepoints**: PELT or Bayesian online detection of mean-shift transitions
3. **model comparison**: fits sigmoid (gradual), step-function (abrupt), and 2-state HMM (unstable), selects by BIC
4. **classification**: combines evidence from sigmoid steepness, changepoint count, Ψ spike, and jump magnitude
5. **persistence**: tests whether post-transition performance holds after perturbation
6. **convergence**: tests whether accuracy, confidence, and RT changepoints align temporally (permutation test)

### validation

synthetic learner recovery (n=100, 20 per type):

| type | recovery rate |
|------|--------------|
| abrupt | 95% |
| gradual | 80% |
| non_learner | 100% |
| unstable | 25% (inherently ambiguous at low switch rates) |
| **overall** | **76%** |

false-aha detection via convergence: 0.38 (false) vs 0.58 (true abrupt).

## cross-domain results

| domain | system | N | classification | Ψ z-score |
|--------|--------|---|---------------|-----------|
| grokking | neural network | 60 runs | dose-response: 0-60% abrupt by WD | 14.3 ± 11.4 |
| mouse maze | rodent | 19 mice | 95% unstable (state-switching) | 3-7 |
| human AGL | human | pending | ? | experiment ready |

### grokking dose-response (60 runs)

| weight decay | grokked | abrupt rate |
|---|---|---|
| 0.00 | 1/10 (10%) | 0% |
| 0.01 | 2/10 (20%) | 0% |
| 0.03 | 10/10 (100%) | 30% |
| 0.10 | 10/10 (100%) | 40% |
| 0.30 | 10/10 (100%) | 0% (too fast to detect) |
| 1.00 | 10/10 (100%) | 60% |

## what's done vs TODO

done:
- [x] inference module (7 files, 1400 lines)
- [x] synthetic validation (76% recovery)
- [x] grokking 60-run sweep with dose-response
- [x] mouse maze reanalysis (19 mice)
- [x] AGL web experiment UI
- [x] AGL analysis adapter (inference-based)
- [x] experiment design with transition-signature predictions
- [x] claim language rewritten (signatures, not phase transitions)

TODO:
- [ ] deploy web experiment (thebeakers.com/study/)
- [ ] collect human data (60-90 students, 25-30 min each)
- [ ] IRB protocol
- [ ] OSF pre-registration

## license

MIT
