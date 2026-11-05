"""Store projections (current + history), resolve players, detect line movement."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from tennis_edge.config import get_settings
from tennis_edge.db.database import Database, utcnow_iso
from tennis_edge.names import NameMatcher


def projection_key(platform: str, market: str, player: str, opponent: str | None, start_time: str) -> str:
    day = str(start_time)[:10]
    return "|".join([platform.lower(), market, player.lower().strip(), (opponent or "").lower().strip(), day])


_matchers: dict[tuple[str, int], NameMatcher] = {}


def get_matcher(db: Database, tour: str) -> NameMatcher:
    n = db.scalar("SELECT COUNT(*) FROM players WHERE tour=?", (tour,)) or 0
    key = (str(db.path) + tour, n)
    if key not in _matchers:
        players = db.query(
            "SELECT p.player_id, p.name_norm, MAX(m.est_match_date) AS last_seen FROM players p "
            "JOIN matches m ON (m.winner_id = p.player_id OR m.loser_id = p.player_id) AND m.est_match_date >= '2022-01-01' "
            "WHERE p.tour = ? GROUP BY p.player_id", (tour,))
        _matchers[key] = NameMatcher(players)
    return _matchers[key]


def resolve_player(db: Database, name: str | None, tour: str | None) -> tuple[str | None, str | None, str]:
    """Return (player_id, tour, how). Tries the given tour, else both."""
    if not name:
        return None, tour, "no name"
    tours = [tour] if tour in ("ATP", "WTA") else ["ATP", "WTA"]
    hits = []
    for t in tours:
        pid, how = get_matcher(db, t).match(name)
        if pid:
            hits.append((pid, t, how))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return None, tour, "name matches players on both tours - specify tour"
    return None, tour, "player not found in historical data"


def store_projections(db: Database, rows: list[dict], source: str) -> dict:
    """Upsert current projections and append history. Returns a summary incl. line moves."""
    now = utcnow_iso()
    inserted = updated = unchanged = 0
    moves, unresolved = [], []
    for r in rows:
        st = r["start_time"].isoformat() if isinstance(r["start_time"], datetime) else str(r["start_time"])
        pid, tour, how_p = resolve_player(db, r["player"], r.get("tour"))
        oid, tour_o, how_o = resolve_player(db, r.get("opponent"), tour or r.get("tour")) if r.get("opponent") else (None, tour, "none")
        tour = tour or tour_o
        if pid is None:
            unresolved.append(f"{r['player']}: {how_p}")
        if r.get("opponent") and oid is None:
            unresolved.append(f"{r['opponent']}: {how_o}")
        key = projection_key(r["platform"], r["market"], r["player"], r.get("opponent"), st)
        payload = json.dumps({k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in r.items()}, default=str)
        h = hashlib.sha1(f"{key}|{r.get('line')}|{r.get('over_odds')}|{r.get('under_odds')}|{now[:16]}".encode()).hexdigest()
        prev = db.query_one("SELECT line, over_odds, under_odds FROM projections_current WHERE projection_key=?", (key,))
        db.execute("INSERT OR IGNORE INTO projection_history(projection_key,platform,market,player_name,opponent_name,"
                   "line,over_odds,under_odds,start_time,source,fetched_at,content_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (key, r["platform"], r["market"], r["player"], r.get("opponent"), r.get("line"), r.get("over_odds"),
                    r.get("under_odds"), st, source, now, h))
        if prev is None:
            db.execute(
                "INSERT INTO projections_current(projection_key,platform,market,tour,player_name,player_id,opponent_name,"
                "opponent_id,line,over_odds,under_odds,payout_structure,start_time,tournament,surface,best_of,source,"
                "fetched_at,first_seen_at,first_line,raw_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, r["platform"], r["market"], tour, r["player"], pid, r.get("opponent"), oid, r.get("line"),
                 r.get("over_odds"), r.get("under_odds"), r.get("payout_structure"), st, r.get("tournament"),
                 r.get("surface"), r.get("best_of"), source, now, now, r.get("line"), payload))
            inserted += 1
            continue
        changed = (prev["line"] != r.get("line") or prev["over_odds"] != r.get("over_odds")
                   or prev["under_odds"] != r.get("under_odds"))
        db.execute("UPDATE projections_current SET line=?, over_odds=?, under_odds=?, fetched_at=?, player_id=?, "
                   "opponent_id=?, tour=?, payout_structure=COALESCE(?, payout_structure), surface=COALESCE(?, surface), "
                   "best_of=COALESCE(?, best_of), tournament=COALESCE(?, tournament), raw_json=? WHERE projection_key=?",
                   (r.get("line"), r.get("over_odds"), r.get("under_odds"), now, pid, oid, tour, r.get("payout_structure"),
                    r.get("surface"), r.get("best_of"), r.get("tournament"), payload, key))
        if changed:
            updated += 1
            if prev["line"] is not None and r.get("line") is not None:
                delta = float(r["line"]) - float(prev["line"])
                if abs(delta) >= get_settings().alert_line_move:
                    moves.append({"projection_key": key, "player": r["player"], "market": r["market"],
                                  "from": prev["line"], "to": r["line"], "delta": delta})
        else:
            unchanged += 1
    db.log_source(source, "import projections", "ok", inserted + updated,
                  f"inserted={inserted} updated={updated} unchanged={unchanged} unresolved={len(unresolved)}")
    db.audit("import_projections", "projections_current", None,
             {"source": source, "inserted": inserted, "updated": updated, "moves": len(moves)})
    return {"inserted": inserted, "updated": updated, "unchanged": unchanged, "line_moves": moves,
            "unresolved": sorted(set(unresolved))}


def line_history(db: Database, key: str) -> list[dict]:
    return db.query("SELECT fetched_at, line, over_odds, under_odds, source FROM projection_history "
                    "WHERE projection_key=? ORDER BY fetched_at", (key,))
