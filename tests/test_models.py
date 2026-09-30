import numpy as np

from tail_risk_model_backtesting_lab.models import fit_all_methods


def test_all_estimators_produce_ordered_finite_tail_risk() -> None:
    losses = np.random.default_rng(11).standard_t(df=5, size=5000)
    estimates = fit_all_methods(losses, confidence=0.99)
    for result in estimates.values():
        assert np.isfinite(result["var"])
        assert np.isfinite(result["expected_shortfall"])
        assert result["expected_shortfall"] > result["var"]


def test_evt_shape_is_positive_for_heavy_tail_sample() -> None:
    losses = np.random.default_rng(13).standard_t(df=3, size=10000)
    result = fit_all_methods(losses)["evt_pot"]
    assert 0.1 < result["shape"] < 0.6
