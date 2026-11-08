"""Exact moments and Monte Carlo estimates for a Bernoulli policy gradient."""

import numpy as np


def bernoulli_gradient_moments(
    p: float, baseline: float = 0.0, n: int = 1
) -> tuple[float, float]:
    """Return mean and variance of mean((A - baseline) * (A - p)).

    Actions are n independent Bernoulli(p) draws. The baseline is fixed with
    respect to these draws and is not differentiated through.
    """
    if not 0 <= p <= 1 or n < 1:
        raise ValueError("Require 0 <= p <= 1 and n >= 1.")
    return p * (1 - p), p * (1 - p) * (1 - p - baseline) ** 2 / n


def compare_bernoulli_baselines(
    p: float, n: int = 32, trials: int = 100_000, seed: int = 0
) -> list[dict[str, str | float]]:
    """Compare exact moments with independent batches, sharing draws across baselines."""
    bernoulli_gradient_moments(p, n=n)
    if trials < 2:
        raise ValueError("Require at least two Monte Carlo trials.")
    # The success count is sufficient; no trials-by-n action array is needed.
    frequencies = np.random.default_rng(seed).binomial(n, p, size=trials) / n
    rows = []
    for name, baseline in [("无 baseline", 0.0), ("总体均值", p), ("最优 baseline", 1 - p)]:
        mean, variance = bernoulli_gradient_moments(p, baseline, n)
        estimates = baseline * p + frequencies * (1 - p - baseline)
        rows.append(
            {
                "方案": name,
                "p": p,
                "b": baseline,
                "理论均值": mean,
                "模拟均值": float(estimates.mean()),
                "理论方差": variance,
                "模拟方差": float(estimates.var(ddof=1)),
            }
        )
    return rows
