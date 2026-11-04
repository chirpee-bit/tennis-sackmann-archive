"""Recommendation tracking, grading, pick'em entries, bankroll and closing lines.

* Recommendations are saved BEFORE the match starts (enforced) with the exact
  line, side, odds or payout-table snapshot and timestamp. Core fields are
  immutable (database trigger).
* ``stake = 0`` means a paper recommendation: tracked for accuracy, no P&L.
* Grading uses verified results only: the historical results database (once a
  match is ingested) or an explicit manual entry, recorded with its source.
* Win / loss / push / void are stored separately; profit is computed from the
  actual odds (single bets) or the pick'em payout table snapshot (entries).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import os

from tennis_edge.betting.evaluator import Opportunity
from tennis_edge.betting.odds import profit_for_outcome
from tennis_edge.betting.pickem import PayoutStructure, load_payouts, settle_entry
from tennis_edge.db.database import Database, utcnow, utcnow_iso

RETIREMENT_RULE = os.environ.get("TE_RETIREMENT_RULE", "void")   # void | settle_as_played (props) - check your platform


class TrackingError(ValueError):
    pass


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    d = datetime.fromisoformat(str(ts))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def save_recommendation(db: Database, opp: Opportunity, stake: float = 0.0, actor: str = "system",
                        allow_no_bet: bool = False, now: datetime | None = None) -> int:
    now = now or utcnow()
    start = _parse(opp.start_time)
    if start is None:
        raise TrackingError("recommendation needs a start time")
    if start <= now:
        raise TrackingError("cannot save a recommendation after the match has started")
    if not opp.is_bet and not allow_no_bet:
        raise TrackingError("opportunity is NO BET: " + "; ".join(opp.no_bet_reasons))
    if stake < 0:
        raise TrackingError("stake must be >= 0")
    payout_snapshot = None
    if opp.payout_structure:
        st = load_payouts().get(opp.payout_structure)
        if st is not None:
            payout_snapshot = json.dumps({"id": st.id, "legs": st.legs, "payouts": st.payouts, "verified": st.verified,
                                          "void_rule": st.void_rule, "family": st.family, "platform": st.platform})
    existing = db.query_one(
        "SELECT id FROM bet_recommendations WHERE platform=? AND market=? AND player_name=? AND "
        "IFNULL(opponent_name,'')=IFNULL(?, '') AND IFNULL(line,-999)=IFNULL(?,-999) AND side=? AND start_time=?",
        (opp.platform, opp.market, opp.player_name, opp.opponent_name, opp.line, opp.side, opp.start_time))
    if existing:
        return int(existing["id"])
    rid = db.execute(
        "INSERT INTO bet_recommendations(created_at,platform,market,tour,player_name,player_id,opponent_name,opponent_id,"
        "line,side,odds_american,payout_structure,payout_snapshot,model_prob,implied_prob,breakeven_prob,"
        "fair_odds_american,edge,ev_per_unit,uncertainty,confidence,rank_score,stake,start_time,model_version,"
        "projection_key,reasons_for,reasons_against) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (now.replace(microsecond=0).isoformat(), opp.platform, opp.market, opp.tour, opp.player_name, opp.player_id,
         opp.opponent_name, opp.opponent_id, opp.line, opp.side, opp.odds_american, opp.payout_structure,
         payout_snapshot, opp.model_prob, opp.implied_prob, opp.breakeven_prob, opp.fair_odds_american, opp.edge,
         opp.ev_per_unit, opp.uncertainty, opp.confidence, opp.rank_score, float(stake), opp.start_time,
         opp.model_version, opp.projection_key, json.dumps(opp.reasons_for), json.dumps(opp.reasons_against)))
    db.audit("save_recommendation", "bet_recommendations", rid,
             {"market": opp.market, "side": opp.side, "line": opp.line, "stake": stake}, actor=actor)
    return int(rid)


def outcome_for(side: str, line: float | None, value: float) -> str:
    if side == "win":
        return "win" if value == 1 else "loss"
    if side == "opp_win":
        return "win" if value == 0 else "loss"
    if line is None:
        raise TrackingError("line required")
    if value == line:
        return "push"
    if side == "over":
        return "win" if value > line else "loss"
    if side == "under":
        return "win" if value < line else "loss"
    raise TrackingError(f"unknown side {side}")


def grade(db: Database, rec_id: int, outcome: str | None = None, result_value: float | None = None,
          source: str = "manual", notes: str | None = None, actor: str = "user") -> dict:
    rec = db.query_one("SELECT * FROM bet_recommendations WHERE id=?", (rec_id,))
    if rec is None:
        raise TrackingError(f"recommendation {rec_id} not found")
    if rec["status"] != "pending":
        raise TrackingError(f"recommendation {rec_id} already graded ({rec['status']})")
    if outcome is None:
        if result_value is None:
            raise TrackingError("give an outcome or a result value")
        outcome = outcome_for(rec["side"], rec["line"], float(result_value))
    if outcome not in ("win", "loss", "push", "void"):
        raise TrackingError("outcome must be win/loss/push/void")
    stake = float(rec["stake"] or 0)
    if stake > 0 and rec["odds_american"] is not None:
        profit = profit_for_outcome(outcome, stake, rec["odds_american"])
    else:
        profit = 0.0     # paper bet, or pick'em leg (P&L realised on the entry)
    now = utcnow_iso()
    db.execute("INSERT INTO settled_results(recommendation_id,outcome,result_value,profit,settled_at,source,notes) "
               "VALUES (?,?,?,?,?,?,?)", (rec_id, outcome, result_value, profit, now, source, notes))
    db.execute("UPDATE bet_recommendations SET status=?, result_value=?, graded_at=?, grade_source=? WHERE id=?",
               (outcome, result_value, now, source, rec_id))
    if profit != 0 or (stake > 0 and rec["odds_american"] is not None):
        db.execute("INSERT INTO bankroll_transactions(ts,kind,amount,reference,note) VALUES (?,?,?,?,?)",
                   (now, "settlement", profit, f"rec:{rec_id}", f"{rec['player_name']} {rec['market']} {outcome}"))
    db.audit("grade", "bet_recommendations", rec_id, {"outcome": outcome, "value": result_value, "source": source},
             actor=actor)
    settle_entries(db)
    return {"id": rec_id, "outcome": outcome, "profit": profit}


# ---------------------------------------------------------------------- results lookup
def result_value_from_match(db: Database, rec: dict) -> tuple[float | None, str, str]:
    """Find the verified historical result for a recommendation. Returns (value, status, source)."""
    start = _parse(rec["start_time"])
    if not rec.get("player_id") or not rec.get("opponent_id") or start is None:
        return None, "unresolved", "missing player ids"
    lo = (start - timedelta(days=10)).date().isoformat()
    hi = (start + timedelta(days=14)).date().isoformat()
    m = db.query_one(
        "SELECT * FROM matches WHERE ((winner_id=? AND loser_id=?) OR (winner_id=? AND loser_id=?)) "
        "AND est_match_date BETWEEN ? AND ? ORDER BY ABS(julianday(est_match_date) - julianday(?)) LIMIT 1",
        (rec["player_id"], rec["opponent_id"], rec["opponent_id"], rec["player_id"], lo, hi, start.date().isoformat()))
    if m is None:
        return None, "not_found", "no verified result yet"
    src = f"historical:{m['source_file']}:{m['match_id']}"
    if m["status"] == "walkover":
        return None, "void", src + " (walkover)"
    player_won = m["winner_id"] == rec["player_id"]
    market = rec["market"]
    if m["status"] in ("retired", "default"):
        if market == "match_winner" or RETIREMENT_RULE == "void":
            return None, "void", src + f" ({m['status']})"
    if m["status"] == "unknown":
        return None, "void", src + " (score not verifiable)"
    if market == "match_winner":
        return (1.0 if player_won else 0.0), "ok", src
    if market == "total_games":
        return m["total_games"], "ok", src
    if market == "first_set_games":
        return m["first_set_games"], ("ok" if m["first_set_games"] is not None else "void"), src
    if market == "total_sets":
        return m["total_sets"], "ok", src
    if market == "player_games_won":
        return (m["winner_games"] if player_won else m["loser_games"]), "ok", src
    stats = {r["player_id"]: r for r in db.query("SELECT * FROM match_stats WHERE match_id=?", (m["match_id"],))}
    me, opp = stats.get(rec["player_id"]), stats.get(rec["opponent_id"])
    if me is None or opp is None:
        return None, "no_stats", src + " (no serve statistics published)"
    if market == "aces":
        return me["aces"], "ok", src
    if market == "double_faults":
        return me["double_faults"], "ok", src
    if market == "break_points_won":
        return (opp["bp_faced"] or 0) - (opp["bp_saved"] or 0), "ok", src
    return None, "unsupported", src


def auto_grade(db: Database, now: datetime | None = None, min_hours_after_start: float = 4) -> dict:
    now = now or utcnow()
    pending = db.query("SELECT * FROM bet_recommendations WHERE status='pending'")
    done, waiting = [], 0
    for rec in pending:
        start = _parse(rec["start_time"])
        if start is None or now < start + timedelta(hours=min_hours_after_start):
            continue
        value, status, src = result_value_from_match(db, rec)
        if status == "ok" and value is not None:
            done.append(grade(db, rec["id"], result_value=float(value), source=src, actor="auto_grader"))
        elif status == "void":
            done.append(grade(db, rec["id"], outcome="void", source=src, actor="auto_grader"))
        else:
            waiting += 1
    db.log_source("auto_grader", "grade", "ok", len(done), f"graded={len(done)} waiting_for_results={waiting}")
    return {"graded": len(done), "waiting": waiting, "results": done}


# ---------------------------------------------------------------------- pick'em entries
def create_entry(db: Database, rec_ids: list[int], structure_id: str, stake: float, actor: str = "user") -> int:
    structures = load_payouts()
    st = structures.get(structure_id)
    if st is None:
        raise TrackingError(f"unknown payout structure {structure_id}")
    if len(rec_ids) != st.legs:
        raise TrackingError(f"{structure_id} needs {st.legs} legs")
    if len(set(rec_ids)) != len(rec_ids):
        raise TrackingError("duplicate legs")
    recs = [db.query_one("SELECT * FROM bet_recommendations WHERE id=?", (i,)) for i in rec_ids]
    if any(r is None for r in recs):
        raise TrackingError("unknown recommendation id")
    if any(r["platform"] != st.platform for r in recs):
        raise TrackingError("all legs must be on the structure's platform")
    if any(_parse(r["start_time"]) <= utcnow() for r in recs):
        raise TrackingError("a leg has already started")
    now = utcnow_iso()
    snap = json.dumps({"id": st.id, "legs": st.legs, "payouts": st.payouts, "void_rule": st.void_rule,
                       "family": st.family, "platform": st.platform, "verified": st.verified})
    eid = db.execute("INSERT INTO pickem_entries(created_at,platform,structure_id,payout_snapshot,stake) VALUES (?,?,?,?,?)",
                     (now, st.platform, st.id, snap, float(stake)))
    db.executemany("INSERT INTO entry_legs(entry_id, recommendation_id) VALUES (?,?)", [(eid, r) for r in rec_ids])
    db.audit("create_entry", "pickem_entries", eid, {"legs": rec_ids, "structure": structure_id, "stake": stake}, actor)
    return int(eid)


def settle_entries(db: Database) -> int:
    n = 0
    structures = load_payouts()
    for e in db.query("SELECT * FROM pickem_entries WHERE status='pending'"):
        legs = db.query("SELECT r.status FROM entry_legs l JOIN bet_recommendations r ON r.id=l.recommendation_id "
                        "WHERE l.entry_id=?", (e["id"],))
        if not legs or any(x["status"] == "pending" for x in legs):
            continue
        snap = json.loads(e["payout_snapshot"])
        st = PayoutStructure(snap["id"], snap["platform"], snap["id"], int(snap["legs"]),
                             {int(k): float(v) for k, v in snap["payouts"].items()}, snap.get("void_rule", "drop_leg"),
                             bool(snap.get("verified")), snap.get("family", ""))
        mult, why = settle_entry([x["status"] for x in legs], st, structures)
        payout = e["stake"] * mult
        profit = payout - e["stake"]
        now = utcnow_iso()
        db.execute("UPDATE pickem_entries SET status='settled', payout=?, profit=?, settled_at=? WHERE id=?",
                   (payout, profit, now, e["id"]))
        db.execute("INSERT INTO bankroll_transactions(ts,kind,amount,reference,note) VALUES (?,?,?,?,?)",
                   (now, "settlement", profit, f"entry:{e['id']}", why))
        db.audit("settle_entry", "pickem_entries", e["id"], {"multiplier": mult, "why": why})
        n += 1
    return n


# ---------------------------------------------------------------------- bankroll and CLV
def add_transaction(db: Database, kind: str, amount: float, note: str = "", actor: str = "user") -> int:
    if kind not in ("deposit", "withdrawal", "adjustment"):
        raise TrackingError("manual transactions must be deposit/withdrawal/adjustment")
    amt = -abs(amount) if kind == "withdrawal" else float(amount)
    tid = db.execute("INSERT INTO bankroll_transactions(ts,kind,amount,reference,note) VALUES (?,?,?,?,?)",
                     (utcnow_iso(), kind, amt, None, note))
    db.audit("bankroll", "bankroll_transactions", tid, {"kind": kind, "amount": amt}, actor)
    return int(tid)


def capture_closing_lines(db: Database, now: datetime | None = None) -> int:
    """Store the last pre-start snapshot of each recommended projection as its closing line."""
    now = now or utcnow()
    n = 0
    for r in db.query("SELECT id, projection_key, start_time FROM bet_recommendations WHERE closing_line IS NULL "
                      "AND projection_key IS NOT NULL"):
        start = _parse(r["start_time"])
        if start is None or start > now:
            continue
        snap = db.query_one("SELECT line, over_odds, under_odds FROM projection_history WHERE projection_key=? "
                            "AND fetched_at <= ? ORDER BY fetched_at DESC LIMIT 1", (r["projection_key"], start.isoformat()))
        if snap:
            rec = db.query_one("SELECT side FROM bet_recommendations WHERE id=?", (r["id"],))
            odds = snap["over_odds"] if rec["side"] in ("over", "win") else snap["under_odds"]
            db.execute("UPDATE bet_recommendations SET closing_line=?, closing_odds=? WHERE id=?", (snap["line"], odds, r["id"]))
            n += 1
    return n
