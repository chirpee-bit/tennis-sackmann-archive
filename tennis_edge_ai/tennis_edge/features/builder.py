"""Build the pre-match feature table for a tour and persist the engine snapshot.

Orientation: Sackmann rows are (winner, loser). Training on that orientation
would leak the outcome, so each match is assigned to (A, B) by a deterministic
hash of its id - roughly half the rows have the winner as B.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_edge.config import get_settings
from tennis_edge.db.database import Database
from tennis_edge.features.engine import FeatureEngine
from tennis_edge.ingest.historical import load_matches_frame
from tennis_edge.tennis import level_group

log = logging.getLogger(__name__)
EPOCH = pd.Timestamp("1970-01-01")

PLAYER_TARGETS = ["ace", "df", "svpt", "SvGms", "bpFaced", "bpSaved"]


def to_t(ts) -> float:
    return (pd.Timestamp(ts) - EPOCH).days


def from_t(t: float) -> pd.Timestamp:
    return EPOCH + pd.Timedelta(days=float(t))


def _flip(match_id: str) -> bool:
    return int(hashlib.md5(match_id.encode()).hexdigest()[:8], 16) % 2 == 1


def run_engine(matches: pd.DataFrame, tour: str, emit_from_year: int = 2005,
               engine: FeatureEngine | None = None) -> tuple[pd.DataFrame, FeatureEngine]:
    """Walk matches chronologically, returning winner/loser-oriented feature rows."""
    engine = engine or FeatureEngine(tour)
    rows = []
    cols = list(matches.columns)
    idx = {c: i for i, c in enumerate(cols)}
    emit_from = to_t(pd.Timestamp(f"{emit_from_year}-01-01"))
    tvals = ((matches["est_match_date"] - EPOCH).dt.days).to_numpy()
    for k, tup in enumerate(matches.itertuples(index=False, name=None)):
        t = float(tvals[k])
        m = {c: tup[idx[c]] for c in cols}
        m["t"] = t
        feats = engine.pre_match(m["winner_id"], m["loser_id"], t, m["surface"],
                                 m.get("winner_rank"), m.get("loser_rank"), m.get("winner_rank_pts"),
                                 m.get("loser_rank_pts"), m.get("winner_age"), m.get("loser_age"))
        if t >= emit_from and m["status"] != "walkover":
            rec = {"match_id": m["match_id"], "t": t, "date": m["est_match_date"], "tour": tour,
                   "surface": m["surface"], "level": m["level"], "round": m["round"], "best_of": m["best_of"],
                   "status": m["status"], "winner_id": m["winner_id"], "loser_id": m["loser_id"],
                   "total_games": m["total_games"], "total_sets": m["total_sets"],
                   "first_set_games": m["first_set_games"], "w_games": m["winner_games"],
                   "l_games": m["loser_games"], "tiebreaks": m["tiebreaks"],
                   "a_ht": m.get("w_ht"), "b_ht": m.get("l_ht"), "a_hand": m.get("w_hand"), "b_hand": m.get("l_hand")}
            for c in PLAYER_TARGETS:
                rec[f"w_{c}"] = m.get(f"w_{c}")
                rec[f"l_{c}"] = m.get(f"l_{c}")
            rec.update(feats)
            rows.append(rec)
        vw = {k2[2:]: v for k2, v in feats.items() if k2.startswith("a_")}
        vl = {k2[2:]: v for k2, v in feats.items() if k2.startswith("b_")}
        engine.update(m, vw, vl)
    return pd.DataFrame(rows), engine


def orient(df: pd.DataFrame) -> pd.DataFrame:
    """Convert winner/loser rows to A/B rows with a hash-random swap. y = 1 if A won."""
    df = df.copy()
    flip = df["match_id"].map(_flip).to_numpy()
    a_cols = [c for c in df.columns if c.startswith("a_")]
    for ca in a_cols:
        cb = "b_" + ca[2:]
        if cb in df.columns:
            av, bv = df[ca].to_numpy(copy=True), df[cb].to_numpy(copy=True)
            df[ca] = np.where(flip, bv, av)
            df[cb] = np.where(flip, av, bv)
    ha, hb = df["h2h_a"].to_numpy(copy=True), df["h2h_b"].to_numpy(copy=True)
    df["h2h_a"], df["h2h_b"] = np.where(flip, hb, ha), np.where(flip, ha, hb)
    df["a_id"] = np.where(flip, df["loser_id"], df["winner_id"])
    df["b_id"] = np.where(flip, df["winner_id"], df["loser_id"])
    df["y"] = np.where(flip, 0, 1)
    df = df.copy()  # de-fragment before adding target columns
    df["a_games"] = np.where(flip, df["l_games"], df["w_games"])
    df["b_games"] = np.where(flip, df["w_games"], df["l_games"])
    new: dict[str, np.ndarray] = {}
    for c in PLAYER_TARGETS:
        wa, la = pd.to_numeric(df[f"w_{c}"], errors="coerce").to_numpy(), pd.to_numeric(df[f"l_{c}"], errors="coerce").to_numpy()
        new[f"a_t_{c}"] = np.where(flip, la, wa)
        new[f"b_t_{c}"] = np.where(flip, wa, la)
    # Break points won by a player = opponent's break points faced - saved
    new["a_t_bpwon"] = new["b_t_bpFaced"] - new["b_t_bpSaved"]
    new["b_t_bpwon"] = new["a_t_bpFaced"] - new["a_t_bpSaved"]
    new["level_group"] = np.array([level_group(t, lv) for t, lv in zip(df["tour"], df["level"])], dtype=object)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    drop = ["winner_id", "loser_id", "w_games", "l_games"] + [f"{p}_{c}" for p in ("w", "l") for c in PLAYER_TARGETS]
    return df.drop(columns=drop)


def build_features(db: Database, tour: str, emit_from_year: int = 2005) -> tuple[pd.DataFrame, FeatureEngine]:
    settings = get_settings()
    matches = load_matches_frame(db, tour)
    if matches.empty:
        raise RuntimeError(f"No {tour} matches in database - run ingestion first")
    raw, engine = run_engine(matches, tour, emit_from_year)
    feats = orient(raw)
    out_dir = settings.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_dir / f"features_{tour}.parquet", index=False)
    save_engine(engine, out_dir / f"engine_{tour}.pkl")
    return feats, engine


def save_engine(engine: FeatureEngine, path: Path, keep_years: float = 4.0) -> None:
    engine.prune(engine.last_t - keep_years * 365.25)
    with open(path, "wb") as fh:
        pickle.dump(engine, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_engine(tour: str) -> FeatureEngine | None:
    path = get_settings().artifacts_dir / f"engine_{tour}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_features(tour: str) -> pd.DataFrame | None:
    path = get_settings().artifacts_dir / f"features_{tour}.parquet"
    return pd.read_parquet(path) if path.exists() else None
