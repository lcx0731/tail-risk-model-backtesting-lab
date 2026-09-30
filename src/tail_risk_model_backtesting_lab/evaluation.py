"""Monte Carlo tail-risk audit and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtesting import evaluate_var
from .distributions import SCENARIOS, simulate_losses, true_risk
from .models import METHODS, fit_all_methods
from .reporting import create_figures


def run_monte_carlo(
    *,
    n_replicates: int = 250,
    n_calibration: int = 2000,
    n_test: int = 5000,
    confidence: float = 0.99,
    seed: int = 20260930,
) -> pd.DataFrame:
    """Fit every model on calibration losses and backtest on independent losses."""

    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        truth = true_risk(scenario, confidence)
        for replicate in range(n_replicates):
            calibration = simulate_losses(scenario, n_calibration, rng)
            test = simulate_losses(scenario, n_test, rng)
            estimates = fit_all_methods(calibration, confidence)
            for method in METHODS:
                estimate = estimates[method]
                backtest = evaluate_var(test, estimate["var"], confidence)
                records.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "method": method,
                        "var_estimate": estimate["var"],
                        "es_estimate": estimate["expected_shortfall"],
                        "evt_shape": estimate["shape"],
                        "true_var": truth["var"],
                        "true_es": truth["expected_shortfall"],
                        **backtest,
                        "kupiec_rejects_5pct": backtest["kupiec_p_value"] < 0.05,
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_replications(replications: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tail-estimate bias and out-of-sample backtest behavior."""

    rows: list[dict[str, object]] = []
    grouped = replications.groupby(["scenario", "method"], sort=False)
    for (scenario, method), group in grouped:
        var_relative_errors = group["var_estimate"] / group["true_var"] - 1.0
        es_relative_errors = group["es_estimate"] / group["true_es"] - 1.0
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "n_replicates": len(group),
                "true_var": float(group["true_var"].iloc[0]),
                "true_es": float(group["true_es"].iloc[0]),
                "mean_var_estimate": float(group["var_estimate"].mean()),
                "var_relative_bias": float(var_relative_errors.mean()),
                "var_relative_bias_mcse": float(var_relative_errors.std(ddof=1) / np.sqrt(len(group))),
                "mean_es_estimate": float(group["es_estimate"].mean()),
                "es_relative_bias": float(es_relative_errors.mean()),
                "es_relative_bias_mcse": float(es_relative_errors.std(ddof=1) / np.sqrt(len(group))),
                "mean_exceedance_rate": float(group["exceedance_rate"].mean()),
                "exceedance_rate_mcse": float(group["exceedance_rate"].std(ddof=1) / np.sqrt(len(group))),
                "mean_pinball_loss": float(group["pinball_loss"].mean()),
                "kupiec_rejection_rate": float(group["kupiec_rejects_5pct"].mean()),
                "mean_evt_shape": float(group["evt_shape"].mean()) if method == "evt_pot" else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_experiment(
    output_dir: str | Path,
    *,
    n_replicates: int = 250,
    n_calibration: int = 2000,
    n_test: int = 5000,
    confidence: float = 0.99,
    seed: int = 20260930,
) -> dict[str, object]:
    """Run the default study and write reproducible tables and figures."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    replications = run_monte_carlo(
        n_replicates=n_replicates,
        n_calibration=n_calibration,
        n_test=n_test,
        confidence=confidence,
        seed=seed,
    )
    summary = summarize_replications(replications)
    summary.to_csv(destination / "simulation_results.csv", index=False, float_format="%.10g")
    create_figures(summary, destination, target_rate=1.0 - confidence)

    result: dict[str, object] = {
        "design": {
            "source": "deterministic synthetic unit-variance Gaussian and Student-t losses",
            "seed": seed,
            "n_replicates_per_scenario": n_replicates,
            "n_calibration": n_calibration,
            "n_test": n_test,
            "confidence": confidence,
            "scenarios": list(SCENARIOS),
            "methods": list(METHODS),
        },
        "results": [
            {
                "scenario": str(row.scenario),
                "method": str(row.method),
                "var_relative_bias": float(row.var_relative_bias),
                "es_relative_bias": float(row.es_relative_bias),
                "mean_exceedance_rate": float(row.mean_exceedance_rate),
                "kupiec_rejection_rate": float(row.kupiec_rejection_rate),
                "mean_pinball_loss": float(row.mean_pinball_loss),
            }
            for row in summary.itertuples(index=False)
        ],
    }
    (destination / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
