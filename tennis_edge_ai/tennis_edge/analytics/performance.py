"""Performance analytics: live tracked results and walk-forward backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_edge.db.database import Database
from tennis_edge.validation.metrics import binary_summary, calibration_table, numeric_summary


def tracked_frame(db: Database) -> pd.DataFrame:
    return db.df("SELECT r.*, s.outcome, s.profit, s.settled_at, s.source AS settle_source FROM bet_recommendations r "
                 "LEFT JOIN settled_results s ON s.recommendation_id = r.id")


def tracked_summary(db: Database) -> dict:
    df = tracked_frame(db)
    entries = db.df("SELECT * FROM pickem_entries")
    out = {"total_recommendations": int(len(df)), "pending": int((df["status"] == "pending").sum()) if len(df) else 0}
    for k in ("win", "loss", "push", "void"):
        out[k] = int((df["status"] == k).sum()) if len(df) else 0
    decided = out["win"] + out["loss"]
    out["win_pct"] = out["win"] / decided if decided else None
    singles = df[(df["stake"] > 0) & df["odds_american"].notna() & df["outcome"].notna()] if len(df) else df
    entry_settled = entries[entries["status"] == "settled"] if len(entries) else entries
    staked = float(singles["stake"].sum() if len(singles) else 0) + float(entry_settled["stake"].sum() if len(entry_settled) else 0)
    profit = float(singles["profit"].sum() if len(singles) else 0) + float(entry_settled["profit"].sum() if len(entry_settled) else 0)
    out["staked"], out["profit"] = staked, profit
    out["roi"] = profit / staked if staked else None
    graded = df[df["status"].isin(["win", "loss"]) & df["model_prob"].notna()] if len(df) else df
    if len(graded):
        y = (graded["status"] == "win").astype(float).to_numpy()
        out["brier"] = float(np.mean((graded["model_prob"].to_numpy() - y) ** 2))
        out["mean_model_prob"] = float(graded["model_prob"].mean())
    else:
        out["brier"] = out["mean_model_prob"] = None
    # closing line value: positive = we beat the closing line
    clv = df[df["closing_line"].notna() & df["line"].notna() & df["side"].isin(["over", "under"])] if len(df) else df
    if len(clv):
        sign = np.where(clv["side"] == "over", 1, -1)
        out["avg_clv_line"] = float(np.mean(sign * (clv["closing_line"] - clv["line"])))
        out["clv_n"] = int(len(clv))
    else:
        out["avg_clv_line"], out["clv_n"] = None, 0
    bank = db.df("SELECT * FROM bankroll_transactions ORDER BY ts")
    out["bankroll_balance"] = float(bank["amount"].sum()) if len(bank) else 0.0
    return out


def tracked_breakdown(db: Database, by: str) -> pd.DataFrame:
    df = tracked_frame(db)
    df = df[df["status"].isin(["win", "loss", "push", "void"])]
    if df.empty:
        return pd.DataFrame()
    if by == "month":
        df = df.assign(month=pd.to_datetime(df["start_time"], utc=True).dt.strftime("%Y-%m"))
    rows = []
    for k, g in df.groupby(by):
        dec = g[g["status"].isin(["win", "loss"])]
        y = (dec["status"] == "win").astype(float)
        staked = g.loc[g["stake"] > 0, "stake"].sum()
        rows.append({by: k, "n": len(g), "wins": int((g.status == "win").sum()), "losses": int((g.status == "loss").sum()),
                     "push": int((g.status == "push").sum()), "void": int((g.status == "void").sum()),
                     "win_pct": float(y.mean()) if len(y) else None,
                     "brier": float(np.mean((dec["model_prob"] - y) ** 2)) if len(dec) else None,
                     "profit": float(g["profit"].fillna(0).sum()),
                     "roi": float(g["profit"].fillna(0).sum() / staked) if staked else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- backtests
def backtest_frame(db: Database, run_ids: list[str] | None = None, market: str | None = None) -> pd.DataFrame:
    where, params = [], []
    if run_ids:
        where.append(f"run_id IN ({','.join('?' * len(run_ids))})")
        params += run_ids
    if market:
        where.append("market = ?")
        params.append(market)
    sql = "SELECT * FROM backtest_predictions" + (" WHERE " + " AND ".join(where) if where else "")
    return db.df(sql, tuple(params))


def latest_runs(db: Database) -> list[str]:
    return [r["value"] for r in db.query("SELECT value FROM meta WHERE key LIKE 'latest_run_%'")]


def backtest_summary(bt: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    if bt.empty:
        return pd.DataFrame()
    for key, g in bt.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        g2 = g[g["prob"].notna()]
        rec = dict(zip(by, key))
        if len(g2):
            rec |= binary_summary(g2["outcome"], g2["prob"])
        if g["pred_mean"].notna().any():
            gg = g[g["pred_mean"].notna() & g["actual"].notna()]
            rec |= {k: v for k, v in numeric_summary(gg["actual"], gg["pred_mean"]).items() if k != "n"}
        rows.append(rec)
    return pd.DataFrame(rows)


def backtest_calibration(bt: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    g = bt[bt["prob"].notna()]
    return calibration_table(g["outcome"].to_numpy(), g["prob"].to_numpy(), bins)


def simulated_flat_roi(bt: pd.DataFrame, vig_american: float = -110, min_edge: float = 0.02) -> pd.DataFrame:
    """Illustrative only: ROI of betting the model side at a fixed -110 price on SYNTHETIC lines.

    Backtest lines are synthetic (rounded baseline means), not real market lines,
    so this is NOT evidence of profitability against real books/platforms.
    """
    from tennis_edge.betting.odds import american_to_decimal, implied_prob_american
    be = implied_prob_american(vig_american)
    d = american_to_decimal(vig_american)
    g = bt[bt["prob"].notna() & (bt["model_name"] != "baseline")].copy()
    if g.empty:
        return pd.DataFrame()
    g["p_side"] = np.maximum(g["prob"], 1 - g["prob"])
    g["side_over"] = g["prob"] >= 0.5
    g = g[g["p_side"] - be >= min_edge]
    win = np.where(g["side_over"], g["outcome"] == 1, g["outcome"] == 0)
    g["pl"] = np.where(win, d - 1, -1.0)
    return g.groupby("market").agg(bets=("pl", "size"), roi=("pl", "mean")).reset_index()
