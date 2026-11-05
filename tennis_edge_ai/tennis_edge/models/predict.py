"""Live predictions for upcoming matches.

Features are produced by the *same* :class:`FeatureEngine` used for training,
restored from the snapshot taken after the last historical match, and evaluated
at the scheduled match date (time-decay applied). Rankings come from the latest
ranking file in the database; age from date of birth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tennis_edge.db.database import Database
from tennis_edge.features.builder import EPOCH, from_t, load_engine
from tennis_edge.models.markov import prob_over
from tennis_edge.models.props import PROP_MARKETS, player_rows
from tennis_edge.models.totals import markov_predict
from tennis_edge.models.training import TourModels, load_tour_models
from tennis_edge.tennis import level_group


@dataclass
class MatchPrediction:
    tour: str
    a_id: str
    b_id: str
    a_name: str
    b_name: str
    when: datetime
    surface: str
    best_of: int
    model_version: str
    p_a_wins: float
    markov: dict
    props: dict                       # market -> {"a": (mu, pmf), "b": (mu, pmf)}
    quality: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ market access
    def distribution(self, market: str, side: str = "a") -> tuple[float, np.ndarray] | None:
        m = self.markov
        if market == "total_games":
            return float(m["total_games_mean"][0]), m["total_games_pmf"][0]
        if market == "first_set_games":
            return float(m["first_set_mean"][0]), m["first_set_pmf"][0]
        if market == "total_sets":
            return float(m["sets_mean"][0]), m["sets_pmf"][0]
        if market == "player_games_won":
            return (float(m["a_games_mean"][0]), m["a_games_pmf"][0]) if side == "a" else \
                (float(m["b_games_mean"][0]), m["b_games_pmf"][0])
        if market in self.props:
            mu, pmf = self.props[market][side]
            return float(mu), pmf
        return None

    def p_over(self, market: str, line: float, side: str = "a") -> tuple[float, float, float] | None:
        d = self.distribution(market, side)
        if d is None:
            return None
        o, u, p = prob_over(d[1][None, :], line)
        return float(o[0]), float(u[0]), float(p[0])


class Predictor:
    def __init__(self, db: Database, tour: str):
        self.db, self.tour = db, tour
        self.engine = load_engine(tour)
        self.models: TourModels | None = load_tour_models(tour)
        self._rank_cache: dict[str, tuple] = {}
        if self.engine is not None:
            self.data_through = from_t(self.engine.last_t).date()
        else:
            self.data_through = None
        latest = db.scalar("SELECT MAX(ranking_date) FROM rankings WHERE tour=?", (tour,))
        self.rank_date = latest
        if latest:
            for r in db.query("SELECT player_id, rank, points FROM rankings WHERE tour=? AND ranking_date=?", (tour, latest)):
                self._rank_cache[r["player_id"]] = (r["rank"], r["points"])

    @property
    def ready(self) -> bool:
        return self.engine is not None and self.models is not None

    def why_not_ready(self) -> str:
        if self.engine is None:
            return f"No feature-engine snapshot for {self.tour} - run scripts/build_features.py"
        if self.models is None:
            return f"No trained models for {self.tour} - run scripts/train_models.py"
        return ""

    def _player(self, pid: str) -> dict:
        return self.db.query_one("SELECT full_name, dob, height_cm, hand FROM players WHERE player_id=?", (pid,)) or {}

    def features(self, a_id: str, b_id: str, when: datetime, surface: str, best_of: int,
                 level: str | None = None) -> tuple[pd.DataFrame, dict]:
        ts = pd.Timestamp(when)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        t = float((ts.normalize() - EPOCH).days)
        pa, pb = self._player(a_id), self._player(b_id)
        ages: list[float | None] = []
        for p in (pa, pb):
            d0 = pd.to_datetime(str(p.get("dob") or "")[:8], format="%Y%m%d", errors="coerce")
            ages.append(None if pd.isna(d0) else (t - (d0 - EPOCH).days) / 365.25)
        ra, rb = self._rank_cache.get(a_id, (None, None)), self._rank_cache.get(b_id, (None, None))
        feats = self.engine.pre_match(a_id, b_id, t, surface, ra[0], rb[0], ra[1], rb[1], ages[0], ages[1])
        row = {"match_id": f"live:{a_id}:{b_id}:{int(t)}", "t": t, "date": from_t(t), "tour": self.tour,
               "surface": surface, "level": level, "best_of": best_of, "status": "scheduled",
               "level_group": level_group(self.tour, level) if level else
               ("Grand Slam" if best_of == 5 else ("ATP 250/500" if self.tour == "ATP" else "WTA 250/500")),
               "a_id": a_id, "b_id": b_id, "a_ht": pa.get("height_cm"), "b_ht": pb.get("height_cm"),
               "a_hand": pa.get("hand"), "b_hand": pb.get("hand")}
        row.update(feats)
        quality = {}
        for side, pid in (("a", a_id), ("b", b_id)):
            st = self.engine.players.get(pid)
            quality[f"{side}_known"] = st is not None
            quality[f"{side}_last_match"] = str(from_t(st.last_t).date()) if st and st.last_t is not None else None
            quality[f"{side}_days_since_last"] = float(t - st.last_t) if st and st.last_t is not None else None
            quality[f"{side}_stat_matches_365"] = feats[f"{side}_stat_matches_365"]
            quality[f"{side}_elo_matches"] = feats[f"{side}_elo_n"]
            quality[f"{side}_ranked"] = self._rank_cache.get(pid) is not None
        quality["data_through"] = str(self.data_through)
        quality["rankings_as_of"] = self.rank_date
        quality["days_beyond_data"] = float(t - self.engine.last_t)
        return pd.DataFrame([row]), quality

    def predict(self, a_id: str, b_id: str, when: datetime, surface: str = "Hard", best_of: int = 3,
                level: str | None = None) -> MatchPrediction:
        if not self.ready:
            raise RuntimeError(self.why_not_ready())
        if a_id == b_id:
            raise ValueError("a player cannot play themselves")
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        df, quality = self.features(a_id, b_id, when, surface, best_of, level)
        p = float(self.models.winner_proba(df)[0])
        mk = markov_predict(df, np.array([p]), self.models.mixture)
        props = {}
        for market in PROP_MARKETS:
            rows = player_rows(df, mk, market)
            mu, pmf = self.models.props[market].predict(rows)
            props[market] = {"a": (mu[0], pmf[0]), "b": (mu[1], pmf[1])}
        names = (self._player(a_id).get("full_name", a_id), self._player(b_id).get("full_name", b_id))
        return MatchPrediction(self.tour, a_id, b_id, names[0], names[1], when, surface, best_of,
                               self.models.version, p, mk, props, quality,
                               {k: v for k, v in df.iloc[0].items() if k.startswith(("a_", "b_", "h2h", "tour_"))})


_predictors: dict[tuple[str, str], Predictor] = {}


def get_predictor(db: Database, tour: str, refresh: bool = False) -> Predictor:
    key = (str(db.path), tour)
    if refresh or key not in _predictors:
        _predictors[key] = Predictor(db, tour)
    return _predictors[key]
