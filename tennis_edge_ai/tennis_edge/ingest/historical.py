"""Normalise historical Sackmann data and load it into SQLite (idempotent)."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tennis_edge.db.database import Database, utcnow_iso
from tennis_edge.ingest.score_parser import parse_score
from tennis_edge.names import normalize_name
from tennis_edge.providers.sackmann import SackmannProvider
from tennis_edge.tennis import ROUND_ORDER, estimate_match_dates, normalize_surface

log = logging.getLogger(__name__)

STAT_MAP = {"ace": "aces", "df": "double_faults", "svpt": "serve_points", "1stIn": "first_in",
            "1stWon": "first_won", "2ndWon": "second_won", "SvGms": "service_games",
            "bpSaved": "bp_saved", "bpFaced": "bp_faced"}


@dataclass
class IngestReport:
    tour: str
    rows_read: int
    duplicates_dropped: int
    invalid_dropped: int
    matches_inserted: int
    stats_inserted: int
    players_upserted: int
    status_counts: dict


def _hash(*parts: object) -> str:
    return hashlib.sha1("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()


def normalize_matches(raw: pd.DataFrame, tour: str) -> tuple[pd.DataFrame, int, int]:
    """Return (clean matches, duplicates dropped, invalid dropped)."""
    df = raw.copy()
    n0 = len(df)
    df = df[df["winner_id"].notna() & df["loser_id"].notna() & df["tourney_date"].notna()]
    df = df[df["winner_id"] != df["loser_id"]]
    invalid = n0 - len(df)
    df["winner_id"] = tour + ":" + df["winner_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["loser_id"] = tour + ":" + df["loser_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce").astype("Int64")
    df = df[df["tourney_date"].notna()]
    df["surface"] = df["surface"].map(normalize_surface)
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)
    df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce").fillna(0).astype(int)
    df["round"] = df["round"].astype(str).str.upper().str.strip()
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(8).astype(int)
    df["draw_size"] = pd.to_numeric(df["draw_size"], errors="coerce")
    df["est_match_date"] = estimate_match_dates(df["tourney_date"], df["round"], df["tourney_level"],
                                                df["best_of"], df["draw_size"])
    df["tourney_key"] = tour + ":" + df["tourney_id"].astype(str)

    # Duplicate-match removal: same event, same pair of players, same round.
    pair = np.where(df["winner_id"] < df["loser_id"], df["winner_id"] + "~" + df["loser_id"],
                    df["loser_id"] + "~" + df["winner_id"])
    df["pair_key"] = pair
    before = len(df)
    df = df.drop_duplicates(subset=["tourney_key", "pair_key", "round"], keep="first")
    dups = before - len(df)

    parsed = df["score"].map(parse_score)
    df["status"] = parsed.map(lambda p: p.status)
    df["total_games"] = parsed.map(lambda p: p.total_games)
    df["total_sets"] = parsed.map(lambda p: p.total_sets)
    df["first_set_games"] = parsed.map(lambda p: p.first_set_games)
    df["winner_games"] = parsed.map(lambda p: p.winner_games)
    df["loser_games"] = parsed.map(lambda p: p.loser_games)
    df["tiebreaks"] = parsed.map(lambda p: p.tiebreaks)
    # Best-of consistency: a "completed" best-of-3 cannot have >3 sets.
    bad = (df["status"] == "completed") & (df["total_sets"].fillna(0) > df["best_of"])
    df.loc[bad, "status"] = "unknown"
    df["match_id"] = (df["tourney_key"] + ":" + df["match_num"].astype(str) + ":" +
                      df["winner_id"].str.split(":").str[1] + ":" + df["loser_id"].str.split(":").str[1])
    df["content_hash"] = [
        _hash(tour, t, p, r, s) for t, p, r, s in zip(df["tourney_key"], df["pair_key"], df["round"], df["score"])
    ]
    df = df.sort_values(["est_match_date", "tourney_date", "tourney_key", "round_order", "match_num"],
                        kind="mergesort").reset_index(drop=True)
    return df, dups, invalid


def _num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ingest_tour(db: Database, tour: str, provider: SackmannProvider | None = None) -> IngestReport:
    provider = provider or SackmannProvider()
    started = utcnow_iso()
    raw = provider.load_matches(tour)
    if raw.empty:
        db.log_source(provider.name, f"ingest {tour}", "unavailable", 0, "no files found", started)
        return IngestReport(tour, 0, 0, 0, 0, 0, 0, {})
    df, dups, invalid = normalize_matches(raw, tour)

    # ---------------- players: from players file, enriched by names seen in matches
    players = provider.load_players(tour)
    prow: dict[str, tuple] = {}
    if not players.empty:
        for r in players.itertuples(index=False):
            pid = f"{tour}:{r.player_id}"
            first = r.name_first if isinstance(r.name_first, str) else ""
            last = r.name_last if isinstance(r.name_last, str) else ""
            full = f"{first} {last}".strip()
            if not full:
                continue
            dob = str(int(r.dob)) if pd.notna(r.dob) else None
            prow[pid] = (pid, tour, str(r.player_id), first, last, full, normalize_name(full),
                         r.hand if isinstance(r.hand, str) else None, dob,
                         r.ioc if isinstance(r.ioc, str) else None, _num(r.height))
    for side in ("winner", "loser"):
        sub = df[[f"{side}_id", f"{side}_name", f"{side}_hand", f"{side}_ioc", f"{side}_ht"]].drop_duplicates(f"{side}_id")
        for pid, name, hand, ioc, ht in sub.itertuples(index=False):
            if pid not in prow and isinstance(name, str):
                prow[pid] = (pid, tour, pid.split(":")[1], None, None, name, normalize_name(name),
                             hand if isinstance(hand, str) else None, None,
                             ioc if isinstance(ioc, str) else None, _num(ht))
    n_players = db.executemany(
        "INSERT INTO players(player_id,tour,source_id,first_name,last_name,full_name,name_norm,hand,dob,ioc,height_cm) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(player_id) DO UPDATE SET full_name=excluded.full_name, "
        "name_norm=excluded.name_norm, hand=COALESCE(excluded.hand, players.hand), "
        "height_cm=COALESCE(excluded.height_cm, players.height_cm)",
        list(prow.values()))

    # ---------------- tournaments
    t = df.drop_duplicates("tourney_key")
    db.executemany(
        "INSERT OR IGNORE INTO tournaments(tourney_key,tour,tourney_id,name,surface,draw_size,level,start_date) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [(k, tour, str(i), n, s, None if pd.isna(d) else int(d), None if pd.isna(lv) else str(lv),
          pd.to_datetime(str(td), format="%Y%m%d").date().isoformat())
         for k, i, n, s, d, lv, td in zip(t["tourney_key"], t["tourney_id"], t["tourney_name"], t["surface"],
                                          t["draw_size"], t["tourney_level"], t["tourney_date"])])

    # ---------------- matches
    mrows = []
    for r in df.itertuples(index=False):
        mrows.append((
            r.match_id, tour, r.tourney_key, pd.to_datetime(str(r.tourney_date), format="%Y%m%d").date().isoformat(),
            r.est_match_date.date().isoformat(), r.round, int(r.round_order), int(r.match_num), int(r.best_of),
            r.surface, None if pd.isna(r.tourney_level) else str(r.tourney_level), r.winner_id, r.loser_id,
            None if pd.isna(r.score) else str(r.score), r.status,
            _num(r.total_games), _num(r.total_sets), _num(r.first_set_games), _num(r.winner_games),
            _num(r.loser_games), int(r.tiebreaks), _num(r.minutes), _num(r.winner_rank), _num(r.loser_rank),
            _num(r.winner_rank_points), _num(r.loser_rank_points), _num(r.winner_age), _num(r.loser_age),
            r.source_file, r.content_hash))
    n_matches = db.executemany(
        "INSERT OR IGNORE INTO matches(match_id,tour,tourney_key,tourney_date,est_match_date,round,round_order,"
        "match_num,best_of,surface,level,winner_id,loser_id,score,status,total_games,total_sets,first_set_games,"
        "winner_games,loser_games,tiebreaks,minutes,winner_rank,loser_rank,winner_rank_pts,loser_rank_pts,"
        "winner_age,loser_age,source_file,content_hash) VALUES (" + ",".join(["?"] * 30) + ")", mrows)

    # ---------------- match statistics (only where the serve stats exist)
    srows = []
    has = df["w_svpt"].notna() & df["l_svpt"].notna() & (pd.to_numeric(df["w_svpt"], errors="coerce") > 0)
    for r in df[has].itertuples(index=False):
        d = r._asdict()
        for prefix, pid, win in (("w_", r.winner_id, 1), ("l_", r.loser_id, 0)):
            srows.append((r.match_id, pid, win) + tuple(_num(d[prefix + k]) for k in STAT_MAP))
    n_stats = db.executemany(
        "INSERT OR IGNORE INTO match_stats(match_id,player_id,is_winner,aces,double_faults,serve_points,first_in,"
        "first_won,second_won,service_games,bp_saved,bp_faced) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", srows)

    status_counts = df["status"].value_counts().to_dict()
    db.log_source(provider.name, f"ingest {tour}", "ok", n_matches,
                  f"read={len(raw)} dup={dups} invalid={invalid} stats_rows={n_stats}", started)
    db.audit("ingest_history", "matches", tour, {"inserted": n_matches, "duplicates": dups})
    return IngestReport(tour, len(raw), dups, invalid, n_matches, n_stats, n_players, status_counts)


def ingest_rankings(db: Database, tour: str, provider: SackmannProvider | None = None) -> int:
    provider = provider or SackmannProvider()
    r = provider.load_rankings(tour)
    if r.empty:
        return 0
    rows = [(tour, pd.to_datetime(str(d), format="%Y%m%d").date().isoformat(), f"{tour}:{p}",
             int(k) if pd.notna(k) else None, _num(pts))
            for d, k, p, pts in zip(r["ranking_date"], r["rank"], r["player"], r["points"])]
    n = db.executemany("INSERT OR IGNORE INTO rankings(tour,ranking_date,player_id,rank,points) VALUES (?,?,?,?,?)", rows)
    db.log_source(provider.name, f"rankings {tour}", "ok", n)
    return n


def load_matches_frame(db: Database, tour: str | None = None) -> pd.DataFrame:
    """Load matches joined with both players' stats in chronological order."""
    where = "WHERE m.tour = ?" if tour else ""
    params = (tour,) if tour else ()
    sql = f"""
    SELECT m.*,
      ws.aces w_ace, ws.double_faults w_df, ws.serve_points w_svpt, ws.first_in w_1stIn, ws.first_won w_1stWon,
      ws.second_won w_2ndWon, ws.service_games w_SvGms, ws.bp_saved w_bpSaved, ws.bp_faced w_bpFaced,
      ls.aces l_ace, ls.double_faults l_df, ls.serve_points l_svpt, ls.first_in l_1stIn, ls.first_won l_1stWon,
      ls.second_won l_2ndWon, ls.service_games l_SvGms, ls.bp_saved l_bpSaved, ls.bp_faced l_bpFaced,
      wp.height_cm w_ht, lp.height_cm l_ht, wp.hand w_hand, lp.hand l_hand
    FROM matches m
    LEFT JOIN match_stats ws ON ws.match_id = m.match_id AND ws.player_id = m.winner_id
    LEFT JOIN match_stats ls ON ls.match_id = m.match_id AND ls.player_id = m.loser_id
    LEFT JOIN players wp ON wp.player_id = m.winner_id
    LEFT JOIN players lp ON lp.player_id = m.loser_id
    {where}
    ORDER BY m.est_match_date, m.tourney_date, m.tourney_key, m.round_order, m.match_num
    """
    df = db.df(sql, params)
    df["est_match_date"] = pd.to_datetime(df["est_match_date"])
    return df
