"""
model_compare.py — compare continuous vs abrupt learning models

for each participant's learning curve, fit three models:
  1. ContinuousModel: smooth sigmoid (gradual improvement)
  2. ChangepointModel: piecewise constant with single step (abrupt jump)
  3. HMMModel: 2-state hidden markov (stochastic switching)

compare by BIC. this replaces ad hoc threshold tests with
principled model selection.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from hmmlearn.hmm import GaussianHMM
import warnings


class ContinuousModel:
    """
    sigmoid learning curve: y(t) = base + (ceiling - base) * sigmoid(rate * (t - midpoint))

    represents gradual, continuous improvement.
    """
    name = "continuous"
    n_params = 4  # base, ceiling, rate, midpoint

    def __init__(self):
        self.params = None
        self.nll = None

    def fit(self, series):
        arr = np.asarray(series, dtype=float)
        T = len(arr)
        t = np.arange(T, dtype=float)

        # initial guesses
        base0 = np.mean(arr[:max(1, T // 5)])
        ceil0 = np.mean(arr[max(1, -T // 5):])
        rate0 = 4.0 / T
        mid0 = T / 2.0

        # fix sigma² at data variance during optimization to stabilize
        # the objective surface. recompute profile likelihood at optimum for BIC.
        sigma2_fixed = float(np.var(arr)) + 1e-10

        def neg_log_lik(params):
            base, ceiling, rate, midpoint = params
            pred = base + (ceiling - base) * expit(rate * (t - midpoint))
            residuals = arr - pred
            # use fixed sigma² for stable optimization landscape
            nll = 0.5 * np.sum(residuals ** 2) / sigma2_fixed
            return nll

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(neg_log_lik, [base0, ceil0, rate0, mid0],
                              method="Nelder-Mead",
                              options={"maxiter": 2000, "xatol": 1e-6})

        self.params = result.x
        # recompute profile NLL at optimum for BIC
        pred = result.x[0] + (result.x[1] - result.x[0]) * expit(result.x[2] * (t - result.x[3]))
        sigma2_profile = float(np.mean((arr - pred) ** 2)) + 1e-10
        self.nll = 0.5 * T * np.log(2 * np.pi * sigma2_profile) + 0.5 * T
        return self

    def predict(self, T):
        base, ceiling, rate, midpoint = self.params
        t = np.arange(T, dtype=float)
        return base + (ceiling - base) * expit(rate * (t - midpoint))

    def bic(self, T):
        return 2 * self.nll + self.n_params * np.log(T)

    @property
    def fitted_rate(self):
        """the sigmoid steepness. high rate = sharp transition."""
        if self.params is None:
            return 0.0
        return abs(self.params[2])

    @property
    def transition_width(self):
        """number of trials for 10%-90% of transition. small = abrupt."""
        rate = self.fitted_rate
        if rate < 1e-6:
            return float("inf")
        # sigmoid goes from 0.1 to 0.9 over a range of ~4.4/rate
        return 4.4 / rate


class ChangepointModel:
    """
    piecewise constant with single changepoint:
      y(t) = mu_before  if t < cp
      y(t) = mu_after   if t >= cp

    represents abrupt transition at a specific trial.
    """
    name = "changepoint"
    n_params = 3  # mu_before, mu_after, changepoint

    def __init__(self):
        self.params = None
        self.nll = None
        self.cp = None

    def fit(self, series):
        arr = np.asarray(series, dtype=float)
        T = len(arr)

        # try every possible changepoint, pick best
        best_nll = np.inf
        best_cp = T // 2
        best_mu = (arr[:T // 2].mean(), arr[T // 2:].mean())

        min_seg = max(3, T // 10)

        for cp in range(min_seg, T - min_seg):
            before = arr[:cp]
            after = arr[cp:]
            mu_b = before.mean()
            mu_a = after.mean()

            resid_b = before - mu_b
            resid_a = after - mu_a
            sigma2 = (np.sum(resid_b ** 2) + np.sum(resid_a ** 2)) / T + 1e-10
            nll = 0.5 * T * np.log(2 * np.pi * sigma2) + 0.5 * T

            if nll < best_nll:
                best_nll = nll
                best_cp = cp
                best_mu = (mu_b, mu_a)

        self.cp = best_cp
        self.params = (best_mu[0], best_mu[1], best_cp)
        self.nll = best_nll
        return self

    def predict(self, T):
        mu_b, mu_a, cp = self.params
        pred = np.full(T, mu_b)
        pred[int(cp):] = mu_a
        return pred

    def bic(self, T):
        return 2 * self.nll + self.n_params * np.log(T)

    @property
    def jump_size(self):
        """magnitude of the step change"""
        if self.params is None:
            return 0.0
        return abs(self.params[1] - self.params[0])


class HMMModel:
    """
    2-state gaussian HMM: models stochastic switching between
    low-performance and high-performance states.

    captures learners who oscillate or have unstable transitions.
    """
    name = "hmm"
    n_params = 6  # 2 means, 2 variances, 2 transition probs (off-diagonal)

    def __init__(self):
        self.model = None
        self.nll = None
        self.states = None

    def fit(self, series):
        arr = np.asarray(series, dtype=float).reshape(-1, 1)
        T = len(arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(n_components=2, covariance_type="diag",
                              n_iter=200, random_state=42, tol=1e-4)
            try:
                hmm.fit(arr)
                log_lik = hmm.score(arr)
                self.model = hmm
                self.nll = -log_lik
                self.states = hmm.predict(arr)
            except Exception:
                # HMM can fail on degenerate data
                self.nll = np.inf
                self.states = np.zeros(T, dtype=int)

        return self

    def predict(self, T):
        if self.model is None:
            return np.zeros(T)
        return self.model.means_[self.states].flatten()[:T]

    def bic(self, T):
        if self.nll == np.inf:
            return np.inf
        return 2 * self.nll + self.n_params * np.log(T)

    @property
    def state_separation(self):
        """distance between the two HMM state means"""
        if self.model is None:
            return 0.0
        means = self.model.means_.flatten()
        return abs(means[1] - means[0])


def fit_and_compare(series, return_models=False):
    """
    fit all three models to a learning curve and compare by BIC.

    args:
        series: 1D array of performance values
        return_models: if True, return fitted model objects too

    returns:
        dict with:
            best_model: name of best model ("continuous", "changepoint", "hmm")
            bic_continuous: BIC for continuous model
            bic_changepoint: BIC for changepoint model
            bic_hmm: BIC for HMM model
            delta_bic: BIC(continuous) - BIC(changepoint)
                       positive = changepoint better, negative = continuous better
            changepoint_idx: index of detected changepoint (if changepoint model wins)
            jump_size: magnitude of step change (if changepoint model wins)
            evidence_strength: "strong" (|ΔBIC|>10), "moderate" (6-10),
                               "weak" (2-6), "inconclusive" (<2)
            models: dict of fitted models (only if return_models=True)
    """
    arr = np.asarray(series, dtype=float)
    T = len(arr)

    m_cont = ContinuousModel().fit(arr)
    m_cp = ChangepointModel().fit(arr)
    m_hmm = HMMModel().fit(arr)

    bics = {
        "continuous": m_cont.bic(T),
        "changepoint": m_cp.bic(T),
        "hmm": m_hmm.bic(T),
    }

    best = min(bics, key=bics.get)
    delta_bic = bics["continuous"] - bics["changepoint"]
    abs_delta = abs(delta_bic)

    if abs_delta > 10:
        strength = "strong"
    elif abs_delta > 6:
        strength = "moderate"
    elif abs_delta > 2:
        strength = "weak"
    else:
        strength = "inconclusive"

    result = {
        "best_model": best,
        "bic_continuous": bics["continuous"],
        "bic_changepoint": bics["changepoint"],
        "bic_hmm": bics["hmm"],
        "delta_bic": delta_bic,
        "changepoint_idx": m_cp.cp,
        "jump_size": m_cp.jump_size,
        "evidence_strength": strength,
        "sigmoid_rate": m_cont.fitted_rate,
        "transition_width": m_cont.transition_width,
    }

    if return_models:
        result["models"] = {
            "continuous": m_cont,
            "changepoint": m_cp,
            "hmm": m_hmm,
        }

    return result
