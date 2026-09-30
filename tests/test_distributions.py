import numpy as np

from tail_risk_model_backtesting_lab.distributions import simulate_losses, true_risk


def test_simulated_losses_are_reproducible_and_standardized() -> None:
    first = simulate_losses("student_t_5", 200000, np.random.default_rng(7))
    second = simulate_losses("student_t_5", 200000, np.random.default_rng(7))
    np.testing.assert_allclose(first, second)
    assert abs(first.mean()) < 0.02
    assert abs(first.std() - 1.0) < 0.03


def test_gaussian_true_risk_matches_known_values() -> None:
    risk = true_risk("gaussian", confidence=0.99)
    assert abs(risk["var"] - 2.32635) < 1e-4
    assert abs(risk["expected_shortfall"] - 2.66521) < 1e-4
