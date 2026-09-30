"""Figures for tail-estimation bias and coverage."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

LABELS = {
    "gaussian": "Gaussian",
    "historical": "Historical",
    "evt_pot": "EVT POT",
}
COLORS = {
    "gaussian": "#718096",
    "historical": "#2b6cb0",
    "evt_pot": "#c05621",
}
SCENARIO_ORDER = ["gaussian", "student_t_10", "student_t_5", "student_t_3"]
SCENARIO_LABELS = ["Gaussian", "t(10)", "t(5)", "t(3)"]


def _bar_chart(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    reference: float = 0.0,
) -> None:
    positions = np.arange(len(SCENARIO_ORDER))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    for method_index, method in enumerate(LABELS):
        subset = summary.loc[summary["method"] == method].set_index("scenario")
        axis.bar(
            positions + (method_index - 1) * width,
            subset.loc[SCENARIO_ORDER, metric],
            width=width,
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.axhline(reference, color="#1a202c", linestyle="--", linewidth=1.3)
    axis.set_xticks(positions, SCENARIO_LABELS)
    axis.set(title=title, ylabel=ylabel)
    axis.legend(frameon=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def create_figures(summary: pd.DataFrame, destination: Path, target_rate: float) -> None:
    """Save VaR bias, ES bias, exceedance, and coverage-rejection charts."""

    plt.style.use("seaborn-v0_8-whitegrid")
    _bar_chart(
        summary,
        "var_relative_bias",
        "Relative bias of 99% VaR estimates",
        "Mean relative bias",
        destination / "var_bias.png",
    )
    _bar_chart(
        summary,
        "es_relative_bias",
        "Relative bias of 99% expected shortfall estimates",
        "Mean relative bias",
        destination / "es_bias.png",
    )
    _bar_chart(
        summary,
        "mean_exceedance_rate",
        "Out-of-sample VaR exceedance rates",
        "Mean exceedance rate",
        destination / "exceedance_rate.png",
        reference=target_rate,
    )
    _bar_chart(
        summary,
        "kupiec_rejection_rate",
        "Frequency of Kupiec coverage rejection at 5%",
        "Rejection rate",
        destination / "kupiec_rejections.png",
        reference=0.05,
    )
