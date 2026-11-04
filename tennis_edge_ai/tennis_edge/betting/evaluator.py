"""Evaluate projections into ranked opportunities - or NO BET.

For every projection the evaluator reports: model probability for each side,
market-implied probability (margin removed when both sides are priced),
pick'em break-even probability (only from a VERIFIED payout table), fair odds,
expected value (only when the payout structure is actually known), projection
difference, a heuristic uncertainty, data freshness, sample size, a confidence
grade and human-readable reasons for and against.

Nothing is invented: if any required input is missing the opportunity is
returned as NO BET with the reason.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from tennis_edge.betting.odds import (american_to_decimal, devig_two_way, ev_per_unit, fair_american,
                                      implied_prob_american)
from tennis_edge.betting.pickem import PayoutStructure, breakeven_leg_prob, leg_equivalent_ev, load_payouts
from tennis_edge.config import get_settings
from tennis_edge.db.database import Database
from tennis_edge.markets import load_markets
from tennis_edge.models.predict import MatchPrediction, get_predictor


@dataclass
class Opportunity:
    projection_key: str
    platform: str
    market: str
    market_label: str
    tour: str | None
    player_name: str
    opponent_name: str | None
    player_id: str | None
    opponent_id: str | None
    line: float | None
    start_time: str | None
    side: str | None = None
    model_prob: float | None = None
    prob_over: float | None = None
    prob_under: float | None = None
    prob_push: float | None = None
    model_mean: float | None = None
    projection_diff: float | None = None
    implied_prob: float | None = None
    implied_note: str = ""
    breakeven_prob: float | None = None
    odds_american: float | None = None
    payout_structure: str | None = None
    payout_verified: bool = False
    fair_odds_american: float | None = None
    ev_per_unit: float | None = None
    ev_note: str = ""
    edge: float | None = None
    uncertainty: float | None = None
    edge_lower: float | None = None
    data_age_days: float | None = None
    projection_age_hours: float | None = None
    sample_size: float | None = None
    confidence: str = "NO BET"
    is_bet: bool = False
    rank_score: float = 0.0
    line_move: float | None = None
    market_verified: bool = False
    model_version: str | None = None
    no_bet_reasons: list[str] = field(default_factory=list)
    reasons_for: list[str] = field(default_factory=list)
    reasons_against: list[str] = field(default_factory=list)
    match_group: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _f(x, nd=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def market_track_record(db: Database) -> dict[str, dict]:
    """Settled-recommendation ROI per market (only counted with >= 30 settled bets)."""
    rows = db.query("SELECT r.market, COUNT(*) n, SUM(s.profit) profit, SUM(r.stake) staked FROM settled_results s "
                    "JOIN bet_recommendations r ON r.id = s.recommendation_id WHERE s.outcome IN ('win','loss') "
                    "GROUP BY r.market")
    return {r["market"]: {"n": r["n"], "roi": (r["profit"] / r["staked"]) if r["staked"] else None} for r in rows}


class Evaluator:
    def __init__(self, db: Database, now: datetime | None = None):
        self.db = db
        self.s = get_settings()
        self.markets = load_markets()
        self.payouts: dict[str, PayoutStructure] = load_payouts()
        self.now = now or datetime.now(timezone.utc)
        self.track = market_track_record(db)
        self._pred_cache: dict[tuple, MatchPrediction] = {}

    # ------------------------------------------------------------------ helpers
    def _structure_for(self, platform: str, sid: str | None) -> PayoutStructure | None:
        if sid and sid in self.payouts:
            return self.payouts[sid]
        verified = [s for s in self.payouts.values() if s.platform == platform and s.verified]
        return min(verified, key=lambda s: s.legs) if verified else None

    def _predict(self, tour, a, b, when, surface, best_of) -> MatchPrediction:
        key = (tour, a, b, when.date().isoformat(), surface, best_of)
        if key not in self._pred_cache:
            self._pred_cache[key] = get_predictor(self.db, tour).predict(a, b, when, surface, best_of)
        return self._pred_cache[key]

    # ------------------------------------------------------------------ main
    def evaluate(self, proj: dict) -> Opportunity:
        m = self.markets.get(proj["market"])
        opp = Opportunity(projection_key=proj["projection_key"], platform=proj["platform"], market=proj["market"],
                          market_label=m.label if m else proj["market"], tour=proj.get("tour"),
                          player_name=proj["player_name"], opponent_name=proj.get("opponent_name"),
                          player_id=proj.get("player_id"), opponent_id=proj.get("opponent_id"),
                          line=proj.get("line"), start_time=proj.get("start_time"))
        no = opp.no_bet_reasons
        if m is None:
            no.append("market not supported by the model")
            return opp
        opp.market_verified = m.verified_for(proj["platform"].split(":")[0] if not proj["platform"].startswith("Sportsbook")
                                             else "Sportsbook")
        if not opp.market_verified:
            opp.reasons_against.append(f"market '{m.label}' not marked as verified for {proj['platform']} (config/markets.json)")
        if proj.get("first_line") is not None and proj.get("line") is not None:
            opp.line_move = float(proj["line"]) - float(proj["first_line"])
        try:
            fetched = datetime.fromisoformat(str(proj.get("fetched_at")))
            opp.projection_age_hours = (self.now - fetched).total_seconds() / 3600
        except (TypeError, ValueError):
            pass
        if not opp.player_id:
            no.append(f"player '{opp.player_name}' not matched to historical data")
        if not opp.opponent_id:
            no.append("opponent unknown or not matched - matchup required to price")
        if not opp.tour:
            no.append("tour (ATP/WTA) unknown")
        try:
            when = datetime.fromisoformat(str(proj["start_time"]))
            when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            no.append("invalid start time")
            return opp
        if when <= self.now:
            no.append("match start time has passed - recommendations are only made before the match")
        if no and (not opp.player_id or not opp.opponent_id or not opp.tour):
            return opp
        pred = get_predictor(self.db, opp.tour)
        if not pred.ready:
            no.append(pred.why_not_ready())
            return opp
        surface = proj.get("surface")
        if not surface:
            surface = "Hard"
            opp.reasons_against.append("surface not supplied - Hard assumed")
        best_of = proj.get("best_of")
        if not best_of:
            best_of = 3
            if opp.tour == "ATP":
                opp.reasons_against.append("best-of not supplied - best of 3 assumed (wrong for ATP Grand Slams)")
        mp = self._predict(opp.tour, opp.player_id, opp.opponent_id, when, surface, int(best_of))
        opp.model_version = mp.model_version
        opp.match_group = "|".join(sorted([opp.player_id, opp.opponent_id])) + "|" + when.date().isoformat()
        q = mp.quality
        player_level = m.player_level
        if player_level:
            opp.sample_size = q["a_stat_matches_365"]
        else:
            opp.sample_size = min(q["a_stat_matches_365"], q["b_stat_matches_365"])
        if proj["market"] == "match_winner":
            opp.sample_size = min(q["a_elo_matches"], q["b_elo_matches"])
        ages = [q.get("a_days_since_last"), q.get("b_days_since_last")]
        opp.data_age_days = max(a for a in ages if a is not None) if any(a is not None for a in ages) else None
        if opp.data_age_days is None or opp.data_age_days > self.s.max_data_age_days:
            no.append(f"player data stale: last recorded match {opp.data_age_days:.0f} days before this match "
                      f"(limit {self.s.max_data_age_days}); historical data runs through {q['data_through']}"
                      if opp.data_age_days is not None else "no recent match history for a player")
        if opp.sample_size is None or opp.sample_size < self.s.min_sample_matches:
            no.append(f"sample too small ({opp.sample_size:.0f} relevant matches < {self.s.min_sample_matches})"
                      if opp.sample_size is not None else "no sample")

        metrics = mp and get_predictor(self.db, opp.tour).models.market_metrics.get(proj["market"], {})
        if proj["market"] != "match_winner":
            if not metrics:
                no.append("no out-of-sample validation for this market")
            elif not metrics.get("beats_baseline_brier", False):
                no.append("model has NOT beaten the naive baseline out-of-sample for this market")
            elif metrics.get("model_line_brier") and metrics.get("base_line_brier"):
                skill = 1 - metrics["model_line_brier"] / metrics["base_line_brier"]
                if skill < self.s.min_market_skill:
                    no.append(f"out-of-sample skill vs baseline only {skill:.2%} (< {self.s.min_market_skill:.1%} "
                              "required) - too weak to bet")
        ece = metrics.get("model_line_ece", metrics.get("ece", 0.03)) if metrics else 0.05

        # ---------------- side probabilities
        sides: dict[str, float] = {}
        if proj["market"] == "match_winner":
            sides = {"win": mp.p_a_wins, "opp_win": 1 - mp.p_a_wins}
            opp.model_mean = mp.p_a_wins
        else:
            if opp.line is None:
                no.append("no line")
                return opp
            res = mp.p_over(proj["market"], float(opp.line), "a")
            if res is None:
                no.append("model cannot price this market")
                return opp
            po, pu, pp = res
            opp.prob_over, opp.prob_under, opp.prob_push = _f(po, 4), _f(pu, 4), _f(pp, 4)
            mean, _ = mp.distribution(proj["market"], "a")
            opp.model_mean = _f(mean, 2)
            sides = {"over": po, "under": pu}
            tm = get_predictor(self.db, opp.tour).models
            if metrics.get("line_calibrated"):
                c_over = float(tm.calibrate_line(proj["market"], np.array([po]))[0])
                sides = {"over": c_over, "under": max(0.0, 1 - c_over - pp)}
                opp.reasons_for.append("probabilities recalibrated with walk-forward isotonic calibration")

        # ---------------- pricing per side
        candidates = []
        both_odds = proj.get("over_odds") is not None and proj.get("under_odds") is not None
        for side, p in sides.items():
            odds = proj.get("over_odds") if side in ("over", "win") else proj.get("under_odds")
            rec = {"side": side, "p": p, "odds": odds, "implied": None, "breakeven": None, "ev": None,
                   "note_i": "", "note_ev": "", "struct": None, "verified": False}
            if odds is not None:
                if both_odds:
                    fair = devig_two_way(proj["over_odds"], proj["under_odds"], "power")
                    rec["implied"] = fair[0] if side in ("over", "win") else fair[1]
                    rec["note_i"] = "two-way margin removed (power method)"
                else:
                    rec["implied"] = implied_prob_american(odds)
                    rec["note_i"] = "single-sided price: includes bookmaker margin"
                push = opp.prob_push or 0.0
                rec["ev"] = ev_per_unit(p, odds, push)
                rec["note_ev"] = "sportsbook EV per 1 unit at the listed price"
                rec["cmp"] = rec["implied"]
            else:
                st = self._structure_for(proj["platform"], proj.get("payout_structure"))
                rec["struct"] = st.id if st else proj.get("payout_structure")
                if st is not None and st.verified:
                    rec["verified"] = True
                    rec["breakeven"] = breakeven_leg_prob(st)
                    rec["ev"] = leg_equivalent_ev(p, st)
                    rec["note_ev"] = (f"pick'em EV of a {st.legs}-leg {st.label} entry IF every leg had this "
                                      f"probability (independent legs)")
                    rec["cmp"] = rec["breakeven"]
                else:
                    rec["cmp"] = None
                    rec["note_ev"] = "EV unavailable: no verified payout structure for this platform"
            rec["edge"] = None if rec["cmp"] is None else p - rec["cmp"]
            candidates.append(rec)
        priced = [c for c in candidates if c["edge"] is not None]
        best = max(priced, key=lambda c: c["edge"]) if priced else max(candidates, key=lambda c: c["p"])
        if not priced:
            no.append(best["note_ev"] if best["note_ev"] else "no price or payout structure to compare against")
        opp.side = best["side"]
        opp.model_prob = _f(best["p"], 4)
        opp.implied_prob = _f(best["implied"], 4)
        opp.implied_note = best["note_i"]
        opp.breakeven_prob = _f(best["breakeven"], 4)
        opp.odds_american = best["odds"]
        opp.payout_structure = best["struct"]
        opp.payout_verified = best["verified"]
        opp.ev_per_unit = _f(best["ev"], 4)
        opp.ev_note = best["note_ev"]
        opp.edge = _f(best["edge"], 4)
        if 0 < best["p"] < 1:
            opp.fair_odds_american = _f(fair_american(best["p"]), 0)
        if opp.line is not None and opp.model_mean is not None and proj["market"] != "match_winner":
            diff = float(opp.model_mean) - float(opp.line)
            opp.projection_diff = _f(diff if opp.side == "over" else -diff, 2)

        # ---------------- uncertainty (heuristic, documented)
        n_eff = max(opp.sample_size or 0, 0)
        p = best["p"]
        opp.uncertainty = _f(math.sqrt(ece ** 2 + p * (1 - p) / (n_eff + 10)), 4)
        if opp.edge is not None:
            opp.edge_lower = _f(opp.edge - opp.uncertainty, 4)

        # ---------------- reasons
        self._reasons(opp, mp, proj)

        # ---------------- confidence and score
        min_edge = self.s.min_edge
        if not no and opp.edge_lower is not None:
            if opp.edge_lower >= 2 * min_edge and (opp.sample_size or 0) >= 20 and (opp.data_age_days or 99) <= 30:
                opp.confidence, opp.is_bet = "A", True
            elif opp.edge_lower >= min_edge:
                opp.confidence, opp.is_bet = "B", True
            elif opp.edge is not None and opp.edge > 0:
                opp.confidence = "LEAN"
                no.append(f"edge lower bound {opp.edge_lower:+.1%} below threshold {min_edge:.1%}")
            else:
                no.append("no positive edge")
        opp.rank_score = self._score(opp, metrics) if opp.is_bet else 0.0
        return opp

    def _score(self, opp: Opportunity, metrics: dict) -> float:
        reliability = math.exp(-(opp.data_age_days or 0) / 90.0) * min(1.0, (opp.sample_size or 0) / 20.0)
        skill = 0.0
        if metrics.get("model_line_brier") and metrics.get("base_line_brier"):
            skill = 1 - metrics["model_line_brier"] / metrics["base_line_brier"]
        elif opp.market == "match_winner" and metrics.get("brier"):
            skill = 1 - metrics["brier"] / 0.25
        market_factor = 1 + float(np.clip(skill * 5, -0.5, 0.5))
        tr = self.track.get(opp.market)
        if tr and tr["n"] >= 30 and tr["roi"] is not None:
            market_factor *= 1 + float(np.clip(tr["roi"], -0.5, 0.5)) * 0.5
        move = 1.0
        if opp.line_move and opp.side in ("over", "under"):
            # market moved toward our side (e.g. line rose and we still like the over) -> mild confirmation
            aligned = (opp.line_move > 0) == (opp.side == "over")
            move = 1.1 if aligned else 0.9
        return round(max(opp.edge_lower or 0, 0) * reliability * market_factor * move * 100, 4)

    def _reasons(self, opp: Opportunity, mp: MatchPrediction, proj: dict) -> None:
        f = mp.features
        a, b = mp.a_name, mp.b_name
        fr, ag = opp.reasons_for, opp.reasons_against
        if opp.projection_diff is not None:
            (fr if opp.projection_diff > 0 else ag).append(
                f"model mean {opp.model_mean} vs line {opp.line} ({opp.projection_diff:+.2f} toward {opp.side})")
        elo_d = f.get("a_elo_blend", 1500) - f.get("b_elo_blend", 1500)
        fr.append(f"surface-blended Elo {a} {f.get('a_elo_blend', 0):.0f} vs {b} {f.get('b_elo_blend', 0):.0f} ({elo_d:+.0f})")
        fr.append(f"serve pts won (adj) {f.get('a_spw_adj', 0):.1%} vs {f.get('b_spw_adj', 0):.1%}; "
                  f"return pts won {f.get('a_rpw_adj', 0):.1%} vs {f.get('b_rpw_adj', 0):.1%}")
        if opp.market in ("aces", "double_faults", "break_points_won"):
            key = {"aces": "ace_rate_surf", "double_faults": "df_rate", "break_points_won": "bp_won_per_rg"}[opp.market]
            okey = {"aces": "ace_allowed_rate", "double_faults": "df_induced_rate", "break_points_won": "bp_faced_per_sg"}[opp.market]
            fr.append(f"{a} rate {f.get('a_' + key, 0):.3f}; opponent allows {f.get('b_' + okey, 0):.3f}")
        h2h = (f.get("h2h_a", 0), f.get("h2h_b", 0))
        if sum(h2h):
            fr.append(f"head-to-head {a} {int(h2h[0])}-{int(h2h[1])} {b}")
        if opp.uncertainty is not None:
            ag.append(f"model uncertainty about +/-{opp.uncertainty:.1%} (heuristic)")
        if (opp.sample_size or 0) < 20:
            ag.append(f"limited sample ({opp.sample_size:.0f} matches)")
        if opp.data_age_days is not None and opp.data_age_days > 30:
            ag.append(f"stale data: {opp.data_age_days:.0f} days since a player's last recorded match")
        if mp.quality.get("days_beyond_data", 0) > 30:
            ag.append(f"historical database ends {mp.quality['data_through']} - recent form/injuries not captured")
        if opp.line_move:
            ag.append(f"line moved {opp.line_move:+.1f} since first seen") if (
                (opp.line_move > 0) != (opp.side == "over")) else fr.append(f"line moved {opp.line_move:+.1f} since first seen")
        if not f.get("a_days_rest") is None and f.get("a_m14", 0) >= 5:
            ag.append(f"{a} heavy recent workload ({f.get('a_m14'):.0f} matches in 14 days)")
        if opp.market == "match_winner":
            ag.append("retirement risk not modelled (settlement rules for retirements vary by platform)")
        else:
            ag.append("if a player retires, platform rules decide void/settle - check your platform")


def rank_opportunities(opps: list[Opportunity]) -> list[Opportunity]:
    return sorted(opps, key=lambda o: (o.is_bet, o.rank_score, o.edge_lower or -1), reverse=True)


def evaluate_current(db: Database, platforms: list[str] | None = None, markets: list[str] | None = None,
                     upcoming_only: bool = True, now: datetime | None = None) -> list[Opportunity]:
    ev = Evaluator(db, now)
    sql = "SELECT * FROM projections_current"
    rows = db.query(sql)
    out = []
    for r in rows:
        if platforms and not any(r["platform"].startswith(p) for p in platforms):
            continue
        if markets and r["market"] not in markets:
            continue
        try:
            out.append(ev.evaluate(r))
        except Exception as exc:  # never let one bad row break the page
            o = Opportunity(r["projection_key"], r["platform"], r["market"], r["market"], r.get("tour"),
                            r["player_name"], r.get("opponent_name"), r.get("player_id"), r.get("opponent_id"),
                            r.get("line"), r.get("start_time"))
            o.no_bet_reasons.append(f"evaluation error: {exc}")
            out.append(o)
    if upcoming_only:
        out = [o for o in out if not any("start time has passed" in x for x in o.no_bet_reasons)]
    # flag correlated legs (same match)
    groups: dict[str, int] = {}
    for o in out:
        if o.is_bet and o.match_group:
            groups[o.match_group] = groups.get(o.match_group, 0) + 1
    for o in out:
        if o.is_bet and o.match_group and groups.get(o.match_group, 0) > 1:
            o.reasons_against.append("other selections from the same match are listed - outcomes are correlated; "
                                     "do not combine them as independent pick'em legs")
    return rank_opportunities(out)
