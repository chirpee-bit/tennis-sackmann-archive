"""Evaluation metrics for probabilistic and numerical forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p > 0.5) == (y > 0.5)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width ECE: sum_b |B|/N * |mean(y_b) - mean(p_b)|."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "mean_pred": p[m].mean(),
                         "observed": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def ranked_probability_score(pmf: np.ndarray, actual: np.ndarray) -> float:
    """Mean RPS of discrete distributions (rows) against integer outcomes."""
    pmf = np.atleast_2d(pmf)
    cdf = np.cumsum(pmf, axis=1)
    k = np.arange(pmf.shape[1])
    obs = (k[None, :] >= np.asarray(actual)[:, None]).astype(float)
    return float(np.mean(np.sum((cdf - obs) ** 2, axis=1)))


def binary_summary(y, p) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "accuracy": accuracy(y, p), "brier": brier(y, p), "log_loss": log_loss(y, p),
            "ece": expected_calibration_error(y, p), "mean_pred": float(p.mean()) if len(p) else float("nan"),
            "base_rate": float(y.mean()) if len(y) else float("nan")}


def numeric_summary(y, yhat) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"n": int(len(y)), "mae": mae(y, yhat), "rmse": rmse(y, yhat), "bias": float(np.mean(yhat - y))}
