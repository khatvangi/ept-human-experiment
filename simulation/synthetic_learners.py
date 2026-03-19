"""
synthetic_learners.py — generate synthetic learning trajectories

creates trajectories with KNOWN ground-truth transition types,
so the inference module can be validated: does it recover what
we put in?

four learner types:
  1. GRADUAL: smooth sigmoid improvement
  2. ABRUPT: flat → sudden jump → plateau (single changepoint)
  3. FALSE_AHA: performance gain without subjective transition marker
  4. NON_LEARNER: stays near chance
  5. UNSTABLE: oscillates between states

each generator produces:
  - accuracy series
  - optional confidence series (correlated with accuracy)
  - optional RT series (inversely correlated with accuracy)
  - optional transfer series (persistent or decaying)
  - ground truth labels
"""

import numpy as np
from scipy.special import expit


def generate_gradual(n_trials=100, base=0.50, ceiling=0.85,
                     rate=0.06, noise=0.10, seed=None):
    """
    smooth sigmoid learner. no abrupt transition.

    accuracy(t) = base + (ceiling - base) * sigmoid(rate * (t - n/2)) + noise
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_trials, dtype=float)
    midpoint = n_trials / 2

    signal = base + (ceiling - base) * expit(rate * (t - midpoint))
    acc = signal + rng.normal(0, noise, n_trials)
    acc = np.clip(acc, 0, 1)

    # confidence tracks accuracy smoothly
    conf = signal * 10  # scale to 1-10
    conf = conf + rng.normal(0, 0.5, n_trials)
    conf = np.clip(conf, 1, 10)

    # RT decreases as accuracy increases (faster when confident)
    rt = 2000 - 800 * signal + rng.normal(0, 150, n_trials)
    rt = np.clip(rt, 300, 5000)

    # transfer: moderate persistence
    transfer_acc = ceiling - 0.10 + rng.normal(0, noise, 30)
    transfer_acc = np.clip(transfer_acc, 0, 1)

    return {
        "accuracy": acc,
        "confidence": conf,
        "rt": rt,
        "transfer": transfer_acc,
        "ground_truth": "gradual",
        "params": {"base": base, "ceiling": ceiling, "rate": rate,
                   "midpoint": midpoint, "noise": noise},
    }


def generate_abrupt(n_trials=100, base=0.50, ceiling=0.90,
                    transition_trial=None, transition_width=3,
                    noise=0.10, seed=None):
    """
    abrupt learner. flat at chance, then sudden jump to high performance.

    accuracy(t) = base + (ceiling - base) * step(t - transition_trial) + noise
    (step is a sharp sigmoid with width=transition_width)
    """
    rng = np.random.default_rng(seed)
    if transition_trial is None:
        transition_trial = n_trials // 2 + rng.integers(-10, 10)

    t = np.arange(n_trials, dtype=float)

    # sharp sigmoid (nearly a step function)
    sharpness = 2.0 / max(transition_width, 0.5)
    signal = base + (ceiling - base) * expit(sharpness * (t - transition_trial))
    acc = signal + rng.normal(0, noise, n_trials)
    acc = np.clip(acc, 0, 1)

    # confidence: jumps at transition
    conf = np.where(t < transition_trial, 3 + rng.normal(0, 0.5, n_trials),
                    8 + rng.normal(0, 0.5, n_trials))
    conf = np.clip(conf, 1, 10)

    # RT: sudden drop at transition
    rt_baseline = 1800 + rng.normal(0, 200, n_trials)
    rt_fast = 900 + rng.normal(0, 100, n_trials)
    rt = np.where(t < transition_trial, rt_baseline, rt_fast)
    rt = np.clip(rt, 300, 5000)

    # transfer: strong persistence
    transfer_acc = ceiling - 0.05 + rng.normal(0, noise * 0.7, 30)
    transfer_acc = np.clip(transfer_acc, 0, 1)

    return {
        "accuracy": acc,
        "confidence": conf,
        "rt": rt,
        "transfer": transfer_acc,
        "ground_truth": "abrupt",
        "params": {"base": base, "ceiling": ceiling,
                   "transition_trial": int(transition_trial),
                   "transition_width": transition_width, "noise": noise},
    }


def generate_false_aha(n_trials=100, base=0.50, ceiling=0.75,
                       noise=0.12, seed=None):
    """
    performance improves gradually, but participant reports sudden insight.
    the "aha" is metacognitive, not behavioral.

    accuracy: gradual sigmoid
    confidence: sudden jump (false signal)
    RT: sudden drop (response style change, not learning)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_trials, dtype=float)
    midpoint = n_trials / 2

    # gradual accuracy
    signal = base + (ceiling - base) * expit(0.04 * (t - midpoint))
    acc = signal + rng.normal(0, noise, n_trials)
    acc = np.clip(acc, 0, 1)

    # false confidence jump (independent of accuracy improvement)
    false_aha_trial = int(midpoint + rng.integers(-5, 15))
    conf = np.where(t < false_aha_trial,
                    3.5 + rng.normal(0, 0.5, n_trials),
                    7.5 + rng.normal(0, 0.5, n_trials))
    conf = np.clip(conf, 1, 10)

    # RT also drops at "aha" (speed-accuracy tradeoff shift, not learning)
    rt = np.where(t < false_aha_trial,
                  1600 + rng.normal(0, 200, n_trials),
                  1000 + rng.normal(0, 150, n_trials))
    rt = np.clip(rt, 300, 5000)

    # transfer: weak (they didn't truly learn the structure)
    transfer_acc = 0.55 + rng.normal(0, noise, 30)
    transfer_acc = np.clip(transfer_acc, 0, 1)

    return {
        "accuracy": acc,
        "confidence": conf,
        "rt": rt,
        "transfer": transfer_acc,
        "ground_truth": "false_aha",
        "params": {"base": base, "ceiling": ceiling, "noise": noise,
                   "false_aha_trial": false_aha_trial},
    }


def generate_non_learner(n_trials=100, base=0.50, noise=0.12, seed=None):
    """
    flat trajectory near chance. no learning.
    """
    rng = np.random.default_rng(seed)

    acc = base + rng.normal(0, noise, n_trials)
    acc = np.clip(acc, 0, 1)

    conf = 3 + rng.normal(0, 1, n_trials)
    conf = np.clip(conf, 1, 10)

    rt = 1500 + rng.normal(0, 300, n_trials)
    rt = np.clip(rt, 300, 5000)

    transfer_acc = base + rng.normal(0, noise, 30)
    transfer_acc = np.clip(transfer_acc, 0, 1)

    return {
        "accuracy": acc,
        "confidence": conf,
        "rt": rt,
        "transfer": transfer_acc,
        "ground_truth": "non_learner",
        "params": {"base": base, "noise": noise},
    }


def generate_unstable(n_trials=100, low=0.45, high=0.85,
                      switch_prob=0.03, noise=0.10, seed=None):
    """
    oscillates between low and high performance states.
    models partial learning with unstable state maintenance.
    """
    rng = np.random.default_rng(seed)

    state = 0  # start in low state
    states = []
    for _ in range(n_trials):
        if rng.random() < switch_prob:
            state = 1 - state
        states.append(state)

    states = np.array(states)
    signal = np.where(states == 0, low, high)
    acc = signal + rng.normal(0, noise, n_trials)
    acc = np.clip(acc, 0, 1)

    conf = signal * 10 + rng.normal(0, 1, n_trials)
    conf = np.clip(conf, 1, 10)

    rt = np.where(states == 0, 1600, 1000) + rng.normal(0, 200, n_trials)
    rt = np.clip(rt, 300, 5000)

    transfer_acc = 0.55 + rng.normal(0, noise, 30)
    transfer_acc = np.clip(transfer_acc, 0, 1)

    return {
        "accuracy": acc,
        "confidence": conf,
        "rt": rt,
        "transfer": transfer_acc,
        "ground_truth": "unstable",
        "params": {"low": low, "high": high, "switch_prob": switch_prob, "noise": noise},
    }


def generate_power_law(n_trials=100, base=0.50, ceiling=0.85,
                       exponent=0.3, noise=0.10, seed=None):
    """
    adversarial: power-law learning curve. NOT in the model family
    (sigmoid, step, HMM). tests whether classifier handles out-of-family data.

    accuracy(t) = base + (ceiling - base) * (t/n)^exponent + noise
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_trials, dtype=float)
    signal = base + (ceiling - base) * (t / n_trials) ** exponent
    acc = signal + rng.normal(0, noise, n_trials)
    acc = np.clip(acc, 0, 1)

    conf = signal * 10 + rng.normal(0, 0.5, n_trials)
    conf = np.clip(conf, 1, 10)

    rt = 2000 - 800 * signal + rng.normal(0, 150, n_trials)
    rt = np.clip(rt, 300, 5000)

    transfer_acc = ceiling - 0.10 + rng.normal(0, noise, 30)
    transfer_acc = np.clip(transfer_acc, 0, 1)

    return {
        "accuracy": acc, "confidence": conf, "rt": rt, "transfer": transfer_acc,
        "ground_truth": "gradual",  # power-law is still gradual learning
        "params": {"base": base, "ceiling": ceiling, "exponent": exponent, "noise": noise},
    }


def generate_double_sigmoid(n_trials=100, base=0.45, mid1=0.65, ceiling=0.90,
                             transition1=30, transition2=70,
                             noise=0.10, seed=None):
    """
    adversarial: two-stage learning (e.g., learn chunks then learn rule).
    NOT in the model family (single sigmoid, single step).
    should ideally be classified as abrupt if both transitions are sharp,
    or gradual if both are slow.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_trials, dtype=float)

    sig1 = expit(0.5 * (t - transition1))
    sig2 = expit(0.5 * (t - transition2))
    signal = base + (mid1 - base) * sig1 + (ceiling - mid1) * sig2
    acc = signal + rng.normal(0, noise, n_trials)
    acc = np.clip(acc, 0, 1)

    conf = signal * 10 + rng.normal(0, 0.5, n_trials)
    conf = np.clip(conf, 1, 10)

    rt = 2000 - 800 * signal + rng.normal(0, 150, n_trials)
    rt = np.clip(rt, 300, 5000)

    transfer_acc = ceiling - 0.05 + rng.normal(0, noise, 30)
    transfer_acc = np.clip(transfer_acc, 0, 1)

    return {
        "accuracy": acc, "confidence": conf, "rt": rt, "transfer": transfer_acc,
        "ground_truth": "gradual",  # two moderate steps = gradual overall (not sharp enough for abrupt)
        "params": {"base": base, "mid1": mid1, "ceiling": ceiling,
                   "t1": transition1, "t2": transition2, "noise": noise},
    }


def generate_cohort(n_per_type=20, n_trials=100, seed=42):
    """
    generate a balanced cohort with known ground truth.
    useful for validating the full inference pipeline.

    returns:
        list of dicts, each with accuracy, confidence, rt, transfer,
        and ground_truth fields.
    """
    rng = np.random.default_rng(seed)
    cohort = []

    generators = [
        ("gradual", generate_gradual),
        ("abrupt", generate_abrupt),
        ("non_learner", generate_non_learner),
        ("unstable", generate_unstable),
        ("false_aha", generate_false_aha),
        ("power_law", generate_power_law),         # adversarial: out-of-model-family
        ("double_sigmoid", generate_double_sigmoid), # adversarial: two-stage learning
    ]

    for type_name, gen_fn in generators:
        for i in range(n_per_type):
            s = int(rng.integers(0, 100000))
            data = gen_fn(n_trials=n_trials, seed=s)
            data["participant_id"] = f"{type_name}_{i:03d}"
            cohort.append(data)

    # shuffle
    rng.shuffle(cohort)
    return cohort
