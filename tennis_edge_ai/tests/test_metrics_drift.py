import numpy as np
import pandas as pd
import pytest

from tennis_edge.validation.drift import performance_drift, psi
from tennis_edge.validation.metrics import (brier, calibration_table, expected_calibration_error, log_loss, mae,
                                            ranked_probability_score, rmse)
from tennis_edge.validation.walk_forward import diebold_mariano_logloss


def test_basic_metrics():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.4])
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))
    assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)
    assert rmse([1, 2, 3], [2, 2, 2]) == pytest.approx(np.sqrt(2 / 3))


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200000)
    y = (rng.uniform(0, 1, len(p)) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 0.5 + 0.5, 0, 1)) > 0.2
    t = calibration_table(y, p)
    assert t["n"].sum() == len(p)


def test_rps_perfect_forecast_zero():
    pmf = np.zeros((2, 5))
    pmf[0, 2] = pmf[1, 4] = 1
    assert ranked_probability_score(pmf, np.array([2, 4])) == pytest.approx(0)


def test_diebold_mariano_sign():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 5000)
    y = (rng.uniform(size=5000) < p).astype(float)
    good = diebold_mariano_logloss(y, p, np.full(5000, 0.5))
    assert good["z"] < -5


def test_psi_and_drift():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.02
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25
    dates = pd.date_range("2024-01-01", periods=2000, freq="D")[:2000]
    p = np.full(2000, 0.6)
    y = (rng.uniform(size=2000) < 0.6).astype(float)
    y[-90:] = 0  # model suddenly wrong in recent window
    d = performance_drift(pd.DataFrame({"match_date": dates, "prob": p, "outcome": y}))
    assert d["status"] == "DRIFT"
