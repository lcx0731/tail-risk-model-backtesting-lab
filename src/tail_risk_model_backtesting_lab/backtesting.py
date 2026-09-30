"""Out-of-sample quantile and coverage diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def kupiec_p_value(
    n_violations: int, n_observations: int, expected_rate: float
) -> float:
    """Likelihood-ratio p-value for unconditional coverage."""

    if not 0 <= n_violations <= n_observations or n_observations <= 0:
        raise ValueError("invalid violation counts")
    if not 0.0 < expected_rate < 1.0:
        raise ValueError("expected_rate must lie strictly between zero and one")
    empirical_rate = n_violations / n_observations
    null_log_likelihood = (
        n_violations * np.log(expected_rate)
        + (n_observations - n_violations) * np.log1p(-expected_rate)
    )
    if n_violations == 0:
        alternative_log_likelihood = 0.0
    elif n_violations == n_observations:
        alternative_log_likelihood = 0.0
    else:
        alternative_log_likelihood = (
            n_violations * np.log(empirical_rate)
            + (n_observations - n_violations) * np.log1p(-empirical_rate)
        )
    statistic = max(0.0, 2.0 * (alternative_log_likelihood - null_log_likelihood))
    return float(chi2.sf(statistic, df=1))


def quantile_loss(losses: np.ndarray, forecast: float, confidence: float) -> float:
    """Mean pinball loss for an upper-tail quantile forecast."""

    residual = np.asarray(losses, dtype=float) - forecast
    values = np.where(
        residual >= 0.0,
        confidence * residual,
        (1.0 - confidence) * (-residual),
    )
    return float(np.mean(values))


def evaluate_var(
    test_losses: np.ndarray,
    var_forecast: float,
    confidence: float = 0.99,
) -> dict[str, float]:
    values = np.asarray(test_losses, dtype=float)
    violations = values > var_forecast
    n_violations = int(np.sum(violations))
    return {
        "exceedance_rate": float(np.mean(violations)),
        "pinball_loss": quantile_loss(values, var_forecast, confidence),
        "kupiec_p_value": kupiec_p_value(
            n_violations, len(values), expected_rate=1.0 - confidence
        ),
    }
