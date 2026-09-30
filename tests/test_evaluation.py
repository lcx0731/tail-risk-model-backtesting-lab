import json

from tail_risk_model_backtesting_lab.evaluation import run_experiment, run_monte_carlo


def test_small_monte_carlo_has_complete_design() -> None:
    results = run_monte_carlo(
        n_replicates=2,
        n_calibration=500,
        n_test=1000,
        seed=17,
    )
    assert len(results) == 2 * 4 * 3
    assert results.groupby(["scenario", "replicate"])["method"].nunique().eq(3).all()


def test_small_end_to_end_run_writes_outputs(tmp_path) -> None:
    summary = run_experiment(
        output_dir=tmp_path,
        n_replicates=2,
        n_calibration=500,
        n_test=1000,
        seed=19,
    )
    assert summary["design"]["n_replicates_per_scenario"] == 2
    json.loads((tmp_path / "run_summary.json").read_text())
    for name in [
        "simulation_results.csv",
        "run_summary.json",
        "var_bias.png",
        "es_bias.png",
        "exceedance_rate.png",
        "kupiec_rejections.png",
    ]:
        assert (tmp_path / name).is_file()
