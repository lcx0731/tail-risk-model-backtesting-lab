"""Unit-variance Gaussian and Student-t loss distributions."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t

SCENARIOS: dict[str, int | None] = {
    "gaussian": None,
    "student_t_10": 10,
    "student_t_5": 5,
    "student_t_3": 3,
}


def simulate_losses(
    scenario: str, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw zero-mean, unit-variance synthetic losses."""

    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {tuple(SCENARIOS)}")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    degrees = SCENARIOS[scenario]
    if degrees is None:
        return rng.normal(size=n_samples)
    scale = np.sqrt((degrees - 2.0) / degrees)
    return scale * rng.standard_t(df=degrees, size=n_samples)


def true_risk(scenario: str, confidence: float = 0.99) -> dict[str, float]:
    """Return the analytical upper-tail VaR and expected shortfall."""

    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {tuple(SCENARIOS)}")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie between 0.5 and 1")
    tail_probability = 1.0 - confidence
    degrees = SCENARIOS[scenario]
    if degrees is None:
        quantile = float(norm.ppf(confidence))
        expected_shortfall = float(norm.pdf(quantile) / tail_probability)
    else:
        raw_quantile = float(t.ppf(confidence, df=degrees))
        scale = np.sqrt((degrees - 2.0) / degrees)
        quantile = scale * raw_quantile
        raw_es = (
            (degrees + raw_quantile**2)
            / (degrees - 1.0)
            * t.pdf(raw_quantile, df=degrees)
            / tail_probability
        )
        expected_shortfall = float(scale * raw_es)
    return {"var": quantile, "expected_shortfall": expected_shortfall}
