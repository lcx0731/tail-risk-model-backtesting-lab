import numpy as np

from tail_risk_model_backtesting_lab.backtesting import (
    evaluate_var,
    kupiec_p_value,
    quantile_loss,
)


def test_kupiec_prefers_expected_coverage() -> None:
    assert kupiec_p_value(10, 1000, 0.01) > 0.9
    assert kupiec_p_value(40, 1000, 0.01) < 1e-6


def test_backtest_metrics_match_hand_calculation() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0])
    result = evaluate_var(losses, var_forecast=2.5, confidence=0.75)
    assert result["exceedance_rate"] == 0.25
    assert result["pinball_loss"] == quantile_loss(losses, 2.5, 0.75)
