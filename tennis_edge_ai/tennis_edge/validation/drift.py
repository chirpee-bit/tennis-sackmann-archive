"""Model drift detection.

* Performance drift: recent-window Brier score compared with the long-run
  out-of-sample Brier using a z-test on per-prediction score differences.
* Feature drift: Population Stability Index (PSI) of key features between the
  training window and the recent window (> 0.25 = significant shift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 50:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def performance_drift(df: pd.DataFrame, date_col: str = "match_date", recent_days: int = 90) -> dict:
    """df needs prob, outcome and a date column. Returns z-score of recent vs reference Brier."""
    d = df[df["prob"].notna()].copy()
    if d.empty:
        return {"status": "no data"}
    d[date_col] = pd.to_datetime(d[date_col])
    cutoff = d[date_col].max() - pd.Timedelta(days=recent_days)
    ref, rec = d[d[date_col] <= cutoff], d[d[date_col] > cutoff]
    if len(ref) < 200 or len(rec) < 50:
        return {"status": "insufficient data", "n_recent": int(len(rec)), "n_ref": int(len(ref))}
    br = (ref["prob"] - ref["outcome"]) ** 2
    bc = (rec["prob"] - rec["outcome"]) ** 2
    se = np.sqrt(br.var(ddof=1) / len(br) + bc.var(ddof=1) / len(bc))
    z = (bc.mean() - br.mean()) / se if se > 0 else 0.0
    return {"status": "DRIFT" if z > 2.5 else ("watch" if z > 1.5 else "stable"), "z": float(z),
            "brier_recent": float(bc.mean()), "brier_reference": float(br.mean()),
            "n_recent": int(len(rec)), "n_ref": int(len(ref)), "recent_from": str(cutoff.date())}


def feature_drift(feats: pd.DataFrame, columns: list[str], recent_days: int = 180) -> pd.DataFrame:
    d = pd.to_datetime(feats["date"])
    cutoff = d.max() - pd.Timedelta(days=recent_days)
    ref = feats[(d <= cutoff) & (d > cutoff - pd.Timedelta(days=3 * 365))]
    rec = feats[d > cutoff]
    rows = []
    for c in columns:
        if c in feats:
            v = psi(pd.to_numeric(ref[c], errors="coerce").to_numpy(), pd.to_numeric(rec[c], errors="coerce").to_numpy())
            rows.append({"feature": c, "psi": v, "flag": "shift" if v > 0.25 else ("moderate" if v > 0.1 else "ok")})
    return pd.DataFrame(rows)
