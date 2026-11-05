"""Chronological walk-forward validation.

For each test year Y:
    train  = eligible matches with date in [Y - window, Y)
    calib  = out-of-sample predictions for year Y-1 from a model trained on
             [Y-1-window, Y-1)  (so the calibrator never sees year Y)
    test   = matches in year Y

No random splits are used anywhere. Features are pre-match by construction
(see ``features/engine.py``); orientation is hash-randomised so the label is not
encoded by column order; duplicate matches were removed at ingestion.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from tennis_edge.models.winner import MODEL_FACTORIES, TemperatureCalibrator

log = logging.getLogger(__name__)

EXCLUDED_LEVEL_GROUPS = {"Futures/ITF", "Other"}


def eligible_winner_rows(feats: pd.DataFrame, min_year: int = 2008) -> pd.DataFrame:
    df = feats[(feats["status"] == "completed") & (~feats["level_group"].isin(EXCLUDED_LEVEL_GROUPS))]
    df = df[pd.to_datetime(df["date"]).dt.year >= min_year]
    return df


def _year(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["date"]).dt.year


def walk_forward_winner(feats: pd.DataFrame, first_test_year: int, window: int = 10,
                        models: tuple[str, ...] = ("elo_baseline", "rank_baseline", "logistic", "lightgbm"),
                        last_year: int | None = None, progress: Callable[[str], None] | None = None
                        ) -> pd.DataFrame:
    df = eligible_winner_rows(feats)
    years = _year(df)
    last_year = last_year or int(years.max())
    out = []
    for Y in range(first_test_year, last_year + 1):
        test = df[years == Y]
        if test.empty:
            continue
        train = df[(years >= Y - window) & (years < Y)]
        prev_train = df[(years >= Y - 1 - window) & (years < Y - 1)]
        prev_test = df[years == Y - 1]
        for name in models:
            model = MODEL_FACTORIES[name]().fit(train, train["y"].to_numpy())
            p = model.predict_proba(test)
            cal = TemperatureCalibrator()
            if name not in ("elo_baseline",) and len(prev_train) > 1000 and len(prev_test) > 200:
                pm = MODEL_FACTORIES[name]().fit(prev_train, prev_train["y"].to_numpy())
                cal.fit(pm.predict_proba(prev_test), prev_test["y"].to_numpy())
            elif name == "elo_baseline" and len(prev_test) > 200:
                cal.fit(model.predict_proba(prev_test), prev_test["y"].to_numpy())
            for variant, probs in ((name, p), (f"{name}_cal", cal.transform(p))):
                out.append(pd.DataFrame({
                    "match_id": test["match_id"].to_numpy(), "date": test["date"].to_numpy(),
                    "tour": test["tour"].to_numpy(), "surface": test["surface"].to_numpy(),
                    "level_group": test["level_group"].to_numpy(), "best_of": test["best_of"].to_numpy(),
                    "model": variant, "prob": probs, "y": test["y"].to_numpy(), "year": Y,
                    "temperature": cal.T if variant.endswith("_cal") else 1.0}))
            if progress:
                progress(f"{Y} {name}: n={len(test)} T={cal.T:.3f}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    from tennis_edge.validation.metrics import binary_summary
    by = by or []
    rows = []
    for key, g in preds.groupby(["model"] + by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(["model"] + by, key)) | binary_summary(g["y"], g["prob"]))
    return pd.DataFrame(rows)


def diebold_mariano_logloss(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    """Paired test on per-match log-loss differences (p1 vs p2). Negative mean = p1 better."""
    p1, p2 = np.clip(p1, 1e-6, 1 - 1e-6), np.clip(p2, 1e-6, 1 - 1e-6)
    l1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    l2 = -(y * np.log(p2) + (1 - y) * np.log(1 - p2))
    d = l1 - l2
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean_diff": float(d.mean()), "se": float(se), "z": float(d.mean() / se) if se > 0 else 0.0}
