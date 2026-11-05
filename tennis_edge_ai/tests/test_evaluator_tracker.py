import json
from datetime import datetime, timedelta, timezone
import sqlite3

import numpy as np
import pytest

from tennis_edge.betting.evaluator import Evaluator, evaluate_current
from tennis_edge.models import predict as predict_mod
from tennis_edge.models.markov import match_distribution
from tennis_edge.models.predict import MatchPrediction
from tennis_edge.models.props import nb_pmf
from tennis_edge.tracking.tracker import (TrackingError, auto_grade, capture_closing_lines, create_entry, grade,
                                          outcome_for, save_recommendation, settle_entries)

NOW = datetime.now(timezone.utc).replace(microsecond=0)
START = (NOW + timedelta(days=1)).isoformat()


class StubModels:
    version = "stub-v1"

    def __init__(self, beats=True):
        self.market_metrics = {m: {"beats_baseline_brier": beats, "model_line_ece": 0.01, "model_line_brier": 0.23,
                                   "base_line_brier": 0.245}
                               for m in ("total_games", "player_games_won", "aces", "first_set_games", "total_sets",
                                         "double_faults", "break_points_won")}
        self.market_metrics["match_winner"] = {"brier": 0.21, "ece": 0.01}

    def calibrate_line(self, market, p):
        return p


class StubPredictor:
    ready = True

    def __init__(self, days_since=5.0, stat_n=200.0, beats=True):
        self.models = StubModels(beats)
        self.days_since, self.stat_n = days_since, stat_n

    def why_not_ready(self):
        return ""

    def predict(self, a, b, when, surface, best_of, level=None):
        md = match_distribution(np.array([0.70]), np.array([0.62]), best_of)
        k = np.arange(md.total_games_pmf.shape[1])
        markov = {"total_games_pmf": md.total_games_pmf, "total_games_mean": md.total_games_pmf @ k,
                  "first_set_pmf": md.first_set_pmf, "first_set_mean": md.first_set_pmf @ np.arange(14),
                  "sets_pmf": md.sets_pmf, "sets_mean": md.sets_pmf @ np.arange(6),
                  "a_games_pmf": md.a_games_pmf, "a_games_mean": md.a_games_pmf @ np.arange(md.a_games_pmf.shape[1]),
                  "b_games_pmf": md.b_games_pmf, "b_games_mean": md.b_games_pmf @ np.arange(md.b_games_pmf.shape[1])}
        props = {"aces": {"a": (9.0, nb_pmf(np.array([9.0]), 0.1)[0]), "b": (4.0, nb_pmf(np.array([4.0]), 0.1)[0])}}
        q = {"a_stat_matches_365": self.stat_n, "b_stat_matches_365": self.stat_n, "a_elo_matches": 200,
             "b_elo_matches": 200, "a_days_since_last": self.days_since, "b_days_since_last": self.days_since,
             "data_through": "2026-10-01", "days_beyond_data": self.days_since}
        f = {"a_elo_blend": 2000, "b_elo_blend": 1900, "a_spw_adj": 0.68, "b_spw_adj": 0.64, "a_rpw_adj": 0.40,
             "b_rpw_adj": 0.37, "h2h_a": 1, "h2h_b": 0, "a_m14": 2, "a_days_rest": 3}
        return MatchPrediction("ATP", a, b, "Ann A", "Bob B", when, surface, best_of, "stub-v1", float(md.p_match[0]),
                               markov, props, q, f)


@pytest.fixture()
def env(tmp_db, tmp_path, monkeypatch):
    payouts = tmp_path / "payouts.json"
    payouts.write_text(json.dumps({"structures": {
        "PP_POWER_2": {"platform": "PrizePicks", "label": "P2", "legs": 2, "payouts": {"2": 3.0}, "verified": True,
                       "family": "PP_POWER"},
        "PP_POWER_3": {"platform": "PrizePicks", "label": "P3", "legs": 3, "payouts": {"3": 5.0}, "verified": True,
                       "family": "PP_POWER"},
        "TG_2": {"platform": "Tailgate", "label": "T2", "legs": 2, "payouts": {"2": 3.0}, "verified": False}}}))
    monkeypatch.setenv("TE_PAYOUTS_FILE", str(payouts))
    from tennis_edge.config import reset_settings_cache
    reset_settings_cache()
    predict_mod._predictors.clear()
    predict_mod._predictors[(str(tmp_db.path), "ATP")] = StubPredictor()
    yield tmp_db
    predict_mod._predictors.clear()


def proj(**kw):
    base = {"projection_key": "k1", "platform": "PrizePicks", "market": "total_games", "tour": "ATP",
            "player_name": "Ann A", "player_id": "ATP:1", "opponent_name": "Bob B", "opponent_id": "ATP:2",
            "line": 20.5, "start_time": START, "surface": "Hard", "best_of": 3, "fetched_at": NOW.isoformat(),
            "first_line": 20.5, "over_odds": None, "under_odds": None, "payout_structure": None}
    base.update(kw)
    return base


def test_pickem_verified_structure_prices_against_breakeven(env):
    o = Evaluator(env, NOW).evaluate(proj())
    assert o.side == "over" and o.model_prob > 0.6
    assert o.breakeven_prob == pytest.approx(3 ** -0.5, abs=1e-4)
    assert o.edge == pytest.approx(o.model_prob - o.breakeven_prob, abs=1e-4)
    assert o.ev_per_unit == pytest.approx(o.model_prob ** 2 * 3 - 1, abs=1e-3)
    assert o.is_bet and o.confidence in ("A", "B")
    assert o.projection_diff > 0


def test_unverified_payout_is_no_bet(env):
    o = Evaluator(env, NOW).evaluate(proj(platform="Tailgate"))
    assert not o.is_bet
    assert o.ev_per_unit is None
    assert any("verified payout" in r for r in o.no_bet_reasons)


def test_sportsbook_two_way_odds_devig_and_ev(env):
    o = Evaluator(env, NOW).evaluate(proj(platform="Sportsbook:BookA", over_odds=-110, under_odds=-110, line=22.5))
    assert o.implied_prob == pytest.approx(0.5, abs=1e-6)
    p = o.model_prob
    assert o.ev_per_unit == pytest.approx(p * (100 / 110) - (1 - p - (o.prob_push or 0)), abs=1e-3)


def test_no_bet_conditions(env):
    ev = Evaluator(env, NOW)
    assert "start time has passed" in " ".join(ev.evaluate(proj(start_time=(NOW - timedelta(hours=1)).isoformat())).no_bet_reasons)
    assert "not matched" in " ".join(ev.evaluate(proj(player_id=None)).no_bet_reasons)
    assert "opponent" in " ".join(ev.evaluate(proj(opponent_id=None)).no_bet_reasons)
    predict_mod._predictors[(str(env.path), "ATP")] = StubPredictor(days_since=400)
    assert "stale" in " ".join(Evaluator(env, NOW).evaluate(proj()).no_bet_reasons)
    predict_mod._predictors[(str(env.path), "ATP")] = StubPredictor(stat_n=2)
    assert "sample too small" in " ".join(Evaluator(env, NOW).evaluate(proj()).no_bet_reasons)
    predict_mod._predictors[(str(env.path), "ATP")] = StubPredictor(beats=False)
    o = Evaluator(env, NOW).evaluate(proj())
    assert not o.is_bet and "NOT beaten the naive baseline" in " ".join(o.no_bet_reasons)


def test_tight_line_is_no_bet(env):
    # line at the model median -> no edge over the 57.7% break-even
    o = Evaluator(env, NOW).evaluate(proj(line=23.5))
    assert not o.is_bet


def test_player_level_market_uses_player_side(env):
    o = Evaluator(env, NOW).evaluate(proj(market="aces", line=6.5))
    assert o.side == "over" and o.model_mean == 9.0


def _insert_players_and_match(db, status="completed", winner="ATP:1", total=21, a_games=12, b_games=9):
    db.execute("INSERT OR IGNORE INTO players(player_id,tour,source_id,full_name,name_norm) VALUES ('ATP:1','ATP','1','Ann A','ann a')")
    db.execute("INSERT OR IGNORE INTO players(player_id,tour,source_id,full_name,name_norm) VALUES ('ATP:2','ATP','2','Bob B','bob b')")
    db.execute("INSERT OR IGNORE INTO tournaments(tourney_key,tour,tourney_id) VALUES ('ATP:T','ATP','T')")
    loser = "ATP:2" if winner == "ATP:1" else "ATP:1"
    day = (NOW + timedelta(days=1)).date().isoformat()
    db.execute("INSERT INTO matches(match_id,tour,tourney_key,tourney_date,est_match_date,winner_id,loser_id,status,"
               "total_games,total_sets,first_set_games,winner_games,loser_games,best_of,surface,content_hash,source_file) "
               "VALUES ('M1','ATP','ATP:T',?,?,?,?,?,?,2,10,?,?,3,'Hard','h1','test.csv')",
               (day, day, winner, loser, status, total, a_games, b_games))
    db.execute("INSERT INTO match_stats(match_id,player_id,is_winner,aces,double_faults,bp_faced,bp_saved) VALUES ('M1','ATP:1',1,7,2,3,2)")
    db.execute("INSERT INTO match_stats(match_id,player_id,is_winner,aces,double_faults,bp_faced,bp_saved) VALUES ('M1','ATP:2',0,3,4,6,3)")


def test_outcome_rules():
    assert outcome_for("over", 20.5, 21) == "win"
    assert outcome_for("over", 21, 21) == "push"
    assert outcome_for("under", 20.5, 21) == "loss"
    assert outcome_for("win", None, 1) == "win"
    assert outcome_for("opp_win", None, 1) == "loss"


def test_tracker_lifecycle_and_auto_grading(env):
    ev = Evaluator(env, NOW)
    o = ev.evaluate(proj(platform="Sportsbook:BookA", over_odds=110, under_odds=-130, line=20.5))
    assert o.is_bet
    rid = save_recommendation(env, o, stake=10, now=NOW)
    assert save_recommendation(env, o, stake=10, now=NOW) == rid           # duplicate -> same record
    with pytest.raises(sqlite3.DatabaseError):
        env.execute("UPDATE bet_recommendations SET line=30 WHERE id=?", (rid,))  # immutable
    with pytest.raises(TrackingError):
        save_recommendation(env, o, now=NOW + timedelta(days=2))             # after start
    _insert_players_and_match(env, total=21)
    res = auto_grade(env, now=NOW + timedelta(days=2))
    assert res["graded"] == 1
    rec = env.query_one("SELECT * FROM bet_recommendations WHERE id=?", (rid,))
    assert rec["status"] == "win" and rec["result_value"] == 21
    settled = env.query_one("SELECT * FROM settled_results WHERE recommendation_id=?", (rid,))
    assert settled["profit"] == pytest.approx(11.0)                         # +110 on 10
    assert env.scalar("SELECT SUM(amount) FROM bankroll_transactions") == pytest.approx(11.0)
    with pytest.raises(TrackingError):
        grade(env, rid, "loss")                                             # cannot regrade


def test_retirement_voids_and_aces_grading(env):
    ev = Evaluator(env, NOW)
    o1 = ev.evaluate(proj(market="aces", line=6.5, projection_key="k2"))
    r1 = save_recommendation(env, o1, now=NOW)
    _insert_players_and_match(env, status="retired")
    auto_grade(env, now=NOW + timedelta(days=2))
    assert env.scalar("SELECT status FROM bet_recommendations WHERE id=?", (r1,)) == "void"


def test_pickem_entry_settlement_with_void(env):
    ev = Evaluator(env, NOW)
    ids = []
    for i, line in enumerate((19.5, 20.5, 18.5)):
        o = ev.evaluate(proj(line=line, projection_key=f"e{i}"))
        assert o.is_bet, o.no_bet_reasons
        ids.append(save_recommendation(env, o, now=NOW))
    eid = create_entry(env, ids, "PP_POWER_3", 10)
    grade(env, ids[0], "win")
    grade(env, ids[1], "void")
    assert env.scalar("SELECT status FROM pickem_entries WHERE id=?", (eid,)) == "pending"
    grade(env, ids[2], "win")
    e = env.query_one("SELECT * FROM pickem_entries WHERE id=?", (eid,))
    assert e["status"] == "settled" and e["payout"] == pytest.approx(30.0) and e["profit"] == pytest.approx(20.0)


def test_closing_line_capture(env):
    o = Evaluator(env, NOW).evaluate(proj(projection_key="clv"))
    rid = save_recommendation(env, o, now=NOW)
    env.execute("INSERT INTO projection_history(projection_key,platform,market,player_name,line,start_time,source,"
                "fetched_at,content_hash) VALUES ('clv','PrizePicks','total_games','Ann A',21.5,?, 't', ?, 'x')",
                (START, (NOW + timedelta(hours=20)).isoformat()))
    assert capture_closing_lines(env, now=NOW + timedelta(days=2)) == 1
    assert env.scalar("SELECT closing_line FROM bet_recommendations WHERE id=?", (rid,)) == 21.5
