"""Gaussian, historical, and peaks-over-threshold tail estimators."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto, norm

METHODS = ("gaussian", "historical", "evt_pot")


def _validate(losses: np.ndarray, confidence: float) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or len(values) < 100 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a finite vector with at least 100 values")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie between 0.5 and 1")
    return values


def fit_gaussian(losses: np.ndarray, confidence: float = 0.99) -> dict[str, float]:
    values = _validate(losses, confidence)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    quantile = float(norm.ppf(confidence))
    return {
        "var": mean + standard_deviation * quantile,
        "expected_shortfall": mean
        + standard_deviation * norm.pdf(quantile) / (1.0 - confidence),
        "shape": float("nan"),
    }


def fit_historical(losses: np.ndarray, confidence: float = 0.99) -> dict[str, float]:
    values = _validate(losses, confidence)
    var = float(np.quantile(values, confidence, method="higher"))
    tail = values[values >= var]
    return {
        "var": var,
        "expected_shortfall": float(np.mean(tail)),
        "shape": float("nan"),
    }


def fit_evt_pot(
    losses: np.ndarray,
    confidence: float = 0.99,
    threshold_quantile: float = 0.90,
) -> dict[str, float]:
    """Fit a generalized Pareto distribution to threshold exceedances."""

    values = _validate(losses, confidence)
    if not 0.5 < threshold_quantile < confidence:
        raise ValueError("threshold_quantile must lie below confidence")
    threshold = float(np.quantile(values, threshold_quantile, method="higher"))
    excesses = values[values > threshold] - threshold
    if len(excesses) < 30:
        raise ValueError("too few threshold exceedances for EVT")
    shape, _, scale = genpareto.fit(excesses, floc=0.0)
    exceedance_probability = len(excesses) / len(values)
    target_tail_probability = 1.0 - confidence
    ratio = exceedance_probability / target_tail_probability
    if abs(shape) < 1e-8:
        var = threshold + scale * np.log(ratio)
    else:
        var = threshold + scale / shape * (ratio**shape - 1.0)
    if shape >= 1.0:
        expected_shortfall = float("inf")
    else:
        expected_shortfall = (var + scale - shape * threshold) / (1.0 - shape)
    return {
        "var": float(var),
        "expected_shortfall": float(expected_shortfall),
        "shape": float(shape),
    }


def fit_all_methods(losses: np.ndarray, confidence: float = 0.99) -> dict[str, dict[str, float]]:
    return {
        "gaussian": fit_gaussian(losses, confidence),
        "historical": fit_historical(losses, confidence),
        "evt_pot": fit_evt_pot(losses, confidence),
    }
