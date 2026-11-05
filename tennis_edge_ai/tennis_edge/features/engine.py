"""Chronological, leakage-free feature engine.

The :class:`FeatureEngine` walks matches strictly in time order. For every match it

1. reads both players' state *as of the match date* (``pre_match``), emitting
   features that use only earlier matches, then
2. folds the match result into the state (``update``).

The same object is pickled after the last historical match and reused to
produce features for upcoming matches, so training and serving features are
computed by identical code.

Components
----------
* Surface-adjusted Elo (FiveThirtyEight-style K schedule, overall + surface
  ratings, blended 50/50).
* Time-decayed serve/return statistics with Bayesian shrinkage to the tour
  surface average, plus opponent-adjusted serve/return residuals.
* Time-decayed form (Elo residual: result minus Elo expectation).
* Surface win rates, H2H, rest/fatigue, age, height, ranking.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tennis_edge.tennis import elo_surface

# ---- stat vector layout --------------------------------------------------------
S_FIELDS = ["w", "svpt", "sv_won", "aces", "df", "first_in", "first_won", "second_won", "sv_gms",
            "bp_faced", "bp_saved", "ret_pts", "ret_won", "aces_allowed", "df_induced", "bp_chances",
            "bp_won", "ret_gms", "spw_resid", "rpw_resid", "games_won", "games_played"]
SI = {k: i for i, k in enumerate(S_FIELDS)}
R_FIELDS = ["n", "wins", "elo_resid"]
RI = {k: i for i, k in enumerate(R_FIELDS)}

STAT_HALFLIFE = 300.0      # days
FORM_HALFLIFE = 150.0
TOUR_HALFLIFE = 730.0
PRIOR_POINTS = 240.0       # shrinkage strength in serve points (~3 matches)
PRIOR_MATCHES = 3.0
SURFACES = ("Hard", "Clay", "Grass")


def _decay(halflife: float, days: float) -> float:
    return 0.5 ** (max(days, 0.0) / halflife)


def elo_k(n_matches: float) -> float:
    return 250.0 / ((n_matches + 5.0) ** 0.4)


def elo_prob(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


@dataclass
class Decayed:
    """A vector of exponentially time-decayed sums."""
    size: int
    halflife: float
    values: np.ndarray = field(default=None)  # type: ignore[assignment]
    t: float = -1e9

    def __post_init__(self):
        if self.values is None:
            self.values = np.zeros(self.size)

    def at(self, t: float) -> np.ndarray:
        return self.values * _decay(self.halflife, t - self.t)

    def add(self, t: float, vec: np.ndarray) -> None:
        self.values = self.at(t) + vec
        self.t = t


@dataclass
class PlayerState:
    elo: float = 1500.0
    n: int = 0
    elo_s: dict = field(default_factory=lambda: {s: 1500.0 for s in SURFACES})
    n_s: dict = field(default_factory=lambda: {s: 0 for s in SURFACES})
    stats: Decayed = field(default_factory=lambda: Decayed(len(S_FIELDS), STAT_HALFLIFE))
    stats_s: dict = field(default_factory=lambda: {s: Decayed(len(S_FIELDS), STAT_HALFLIFE) for s in SURFACES})
    res: Decayed = field(default_factory=lambda: Decayed(len(R_FIELDS), FORM_HALFLIFE))
    res_s: dict = field(default_factory=lambda: {s: Decayed(len(R_FIELDS), STAT_HALFLIFE) for s in SURFACES})
    recent: deque = field(default_factory=lambda: deque(maxlen=12))   # (t, minutes)
    last_t: float | None = None
    last_rank: float | None = None
    last_rank_pts: float | None = None
    last_age: float | None = None
    last_age_t: float | None = None
    stat_matches: list = field(default_factory=list)                 # t of matches with stats (last 60)


def _ratio(num: float, den: float, prior: float, k: float) -> float:
    return (num + prior * k) / (den + k)


class FeatureEngine:
    def __init__(self, tour: str):
        self.tour = tour
        self.players: dict[str, PlayerState] = {}
        self.h2h: dict[tuple[str, str], list[int]] = {}
        # Tour averages per surface: [svpt, sv_won, aces, df, ret_pts..] decayed
        self.tour_avg = {s: Decayed(4, TOUR_HALFLIFE) for s in SURFACES}
        self.last_t: float = 0.0
        self.matches_seen = 0

    # ------------------------------------------------------------------ helpers
    def _p(self, pid: str) -> PlayerState:
        st = self.players.get(pid)
        if st is None:
            st = PlayerState()
            self.players[pid] = st
        return st

    def tour_rates(self, surface: str, t: float) -> dict[str, float]:
        v = self.tour_avg[surface].at(t)
        if v[0] < 500:  # not enough history yet -> generic professional averages
            return {"spw": 0.62 if self.tour == "ATP" else 0.56, "ace": 0.07 if self.tour == "ATP" else 0.035,
                    "df": 0.035 if self.tour == "ATP" else 0.045}
        return {"spw": v[1] / v[0], "ace": v[2] / v[0], "df": v[3] / v[0]}

    # ------------------------------------------------------------------ player view
    def player_view(self, pid: str, t: float, surface: str) -> dict[str, float]:
        st = self.players.get(pid) or PlayerState()
        surf = elo_surface(surface)
        surf = surf if surf in SURFACES else "Hard"
        tr = self.tour_rates(surf, t)
        f_avg, g_avg = tr["spw"], 1 - tr["spw"]
        s = st.stats.at(t)
        ss = st.stats_s[surf].at(t)
        r = st.res.at(t)
        rs = st.res_s[surf].at(t)
        k = PRIOR_POINTS
        out: dict[str, float] = {}
        out["elo"] = st.elo
        out["elo_surf"] = st.elo_s[surf]
        out["elo_n"] = float(st.n)
        out["elo_surf_n"] = float(st.n_s[surf])
        # blend surface and overall Elo, trusting surface rating more with experience
        wsurf = min(st.n_s[surf], 30) / 60.0
        out["elo_blend"] = (1 - wsurf) * st.elo + wsurf * st.elo_s[surf]
        out["stat_w"] = s[SI["w"]]
        out["stat_w_surf"] = ss[SI["w"]]
        # raw (shrunk) serve/return rates
        out["spw"] = _ratio(s[SI["sv_won"]], s[SI["svpt"]], f_avg, k)
        out["rpw"] = _ratio(s[SI["ret_won"]], s[SI["ret_pts"]], g_avg, k)
        out["spw_surf"] = _ratio(ss[SI["sv_won"]], ss[SI["svpt"]], out["spw"], k)
        out["rpw_surf"] = _ratio(ss[SI["ret_won"]], ss[SI["ret_pts"]], out["rpw"], k)
        out["spw_adj"] = f_avg + s[SI["spw_resid"]] / (s[SI["w"]] + PRIOR_MATCHES)
        out["rpw_adj"] = g_avg + s[SI["rpw_resid"]] / (s[SI["w"]] + PRIOR_MATCHES)
        out["ace_rate"] = _ratio(s[SI["aces"]], s[SI["svpt"]], tr["ace"], k)
        out["ace_rate_surf"] = _ratio(ss[SI["aces"]], ss[SI["svpt"]], out["ace_rate"], k)
        out["df_rate"] = _ratio(s[SI["df"]], s[SI["svpt"]], tr["df"], k)
        out["ace_allowed_rate"] = _ratio(s[SI["aces_allowed"]], s[SI["ret_pts"]], tr["ace"], k)
        out["df_induced_rate"] = _ratio(s[SI["df_induced"]], s[SI["ret_pts"]], tr["df"], k)
        out["first_in"] = _ratio(s[SI["first_in"]], s[SI["svpt"]], 0.62, k)
        out["first_won"] = _ratio(s[SI["first_won"]], s[SI["first_in"]], 0.72 if self.tour == "ATP" else 0.64, k * 0.6)
        second = s[SI["svpt"]] - s[SI["first_in"]]
        out["second_won"] = _ratio(s[SI["second_won"]], second, 0.51 if self.tour == "ATP" else 0.46, k * 0.4)
        out["bp_save"] = _ratio(s[SI["bp_saved"]], s[SI["bp_faced"]], 0.62 if self.tour == "ATP" else 0.55, 15)
        out["bp_conv"] = _ratio(s[SI["bp_won"]], s[SI["bp_chances"]], 0.38 if self.tour == "ATP" else 0.45, 15)
        out["bp_chances_per_rg"] = _ratio(s[SI["bp_chances"]], s[SI["ret_gms"]], 0.6 if self.tour == "ATP" else 0.85, 20)
        out["bp_won_per_rg"] = _ratio(s[SI["bp_won"]], s[SI["ret_gms"]], 0.22 if self.tour == "ATP" else 0.36, 20)
        out["bp_faced_per_sg"] = _ratio(s[SI["bp_faced"]], s[SI["sv_gms"]], 0.6 if self.tour == "ATP" else 0.85, 20)
        out["games_share"] = _ratio(s[SI["games_won"]], s[SI["games_played"]], 0.5, 30)
        # results / form
        out["form"] = r[RI["elo_resid"]] / (r[RI["n"]] + PRIOR_MATCHES)
        out["win_rate"] = _ratio(r[RI["wins"]], r[RI["n"]], 0.5, PRIOR_MATCHES)
        out["form_n"] = r[RI["n"]]
        out["surf_win_rate"] = _ratio(rs[RI["wins"]], rs[RI["n"]], 0.5, PRIOR_MATCHES * 2)
        out["surf_n"] = rs[RI["n"]]
        # fatigue / rest
        if st.last_t is None:
            out["days_rest"] = 60.0
        else:
            out["days_rest"] = min(max(t - st.last_t, 0.0), 60.0)
        out["m7"] = float(sum(1 for (tt, _) in st.recent if 0 <= t - tt <= 7))
        out["m14"] = float(sum(1 for (tt, _) in st.recent if 0 <= t - tt <= 14))
        out["min14"] = float(sum((mm or 0.0) for (tt, mm) in st.recent if 0 <= t - tt <= 14))
        out["stat_matches_365"] = float(sum(1 for tt in st.stat_matches if 0 <= t - tt <= 365))
        out["rank"] = st.last_rank if st.last_rank is not None else np.nan
        out["rank_pts"] = st.last_rank_pts if st.last_rank_pts is not None else np.nan
        out["age"] = (st.last_age + (t - st.last_age_t) / 365.25) if st.last_age is not None else np.nan
        out["days_since_seen"] = (t - st.last_t) if st.last_t is not None else np.nan
        return out

    def h2h_view(self, a: str, b: str) -> tuple[float, float]:
        key = (a, b) if a < b else (b, a)
        rec = self.h2h.get(key)
        if not rec:
            return 0.0, 0.0
        wa = rec[0] if a < b else rec[1]
        wb = rec[1] if a < b else rec[0]
        return float(wa), float(wb)

    def pre_match(self, a: str, b: str, t: float, surface: str, rank_a=None, rank_b=None,
                  pts_a=None, pts_b=None, age_a=None, age_b=None) -> dict[str, float]:
        va = self.player_view(a, t, surface)
        vb = self.player_view(b, t, surface)
        # match-row info (known pre-match: rank at tournament start, age)
        for v, rk, pt, ag in ((va, rank_a, pts_a, age_a), (vb, rank_b, pts_b, age_b)):
            if rk is not None and not (isinstance(rk, float) and math.isnan(rk)):
                v["rank"] = float(rk)
            if pt is not None and not (isinstance(pt, float) and math.isnan(pt)):
                v["rank_pts"] = float(pt)
            if ag is not None and not (isinstance(ag, float) and math.isnan(ag)):
                v["age"] = float(ag)
        feats = {f"a_{k}": v for k, v in va.items()}
        feats.update({f"b_{k}": v for k, v in vb.items()})
        ha, hb = self.h2h_view(a, b)
        feats["h2h_a"], feats["h2h_b"] = ha, hb
        surf = elo_surface(surface)
        feats["tour_spw"] = self.tour_rates(surf if surf in SURFACES else "Hard", t)["spw"]
        feats["tour_ace"] = self.tour_rates(surf if surf in SURFACES else "Hard", t)["ace"]
        feats["tour_df"] = self.tour_rates(surf if surf in SURFACES else "Hard", t)["df"]
        return feats

    # ------------------------------------------------------------------ update
    def update(self, m: dict, view_w: dict | None = None, view_l: dict | None = None) -> None:
        """Fold one finished match into the state. ``m`` uses Sackmann column names.

        ``view_w``/``view_l`` may pass the pre-match player views already computed
        for this match (pure optimisation - identical values)."""
        status = m.get("status")
        if status == "walkover":
            return
        t = float(m["t"])
        w, l = m["winner_id"], m["loser_id"]
        sw, sl = self._p(w), self._p(l)
        surf = elo_surface(m.get("surface") or "Hard")
        surf = surf if surf in SURFACES else "Hard"

        # --- Elo
        p_all = elo_prob(sw.elo, sl.elo)
        p_surf = elo_prob(sw.elo_s[surf], sl.elo_s[surf])
        mult = 1.0 if status == "completed" else 0.5
        if m.get("level") == "G":
            mult *= 1.1
        kw, kl = elo_k(sw.n) * mult, elo_k(sl.n) * mult
        sw.elo += kw * (1 - p_all)
        sl.elo -= kl * (1 - p_all)
        kws, kls = elo_k(sw.n_s[surf]) * mult, elo_k(sl.n_s[surf]) * mult
        sw.elo_s[surf] += kws * (1 - p_surf)
        sl.elo_s[surf] -= kls * (1 - p_surf)
        sw.n += 1
        sl.n += 1
        sw.n_s[surf] += 1
        sl.n_s[surf] += 1

        # --- results / form (expected uses pre-update blended prob)
        for st, won, pe in ((sw, 1.0, p_all), (sl, 0.0, 1 - p_all)):
            vec = np.zeros(len(R_FIELDS))
            vec[RI["n"]] = 1.0
            vec[RI["wins"]] = won
            vec[RI["elo_resid"]] = won - pe
            st.res.add(t, vec)
            st.res_s[surf].add(t, vec)

        # --- serve / return stats
        def g(key):
            v = m.get(key)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return None if math.isnan(v) else v

        w_svpt, l_svpt = g("w_svpt"), g("l_svpt")
        if w_svpt and l_svpt and w_svpt > 0 and l_svpt > 0 and status in ("completed", "retired"):
            tr = self.tour_rates(surf, t)
            f_avg, g_avg = tr["spw"], 1 - tr["spw"]
            # opponents' pre-match return/serve skill for residuals
            vw = view_w if view_w is not None else self.player_view(w, t, surf)
            vl = view_l if view_l is not None else self.player_view(l, t, surf)
            for me, opp, mp, op, mv, ov in (("w_", "l_", sw, sl, vw, vl), ("l_", "w_", sl, sw, vl, vw)):
                svpt, osvpt = g(me + "svpt"), g(opp + "svpt")
                fw, sw2 = g(me + "1stWon") or 0.0, g(me + "2ndWon") or 0.0
                ofw, osw2 = g(opp + "1stWon") or 0.0, g(opp + "2ndWon") or 0.0
                sv_won = fw + sw2
                ret_won = osvpt - (ofw + osw2)
                if sv_won > svpt or ret_won < 0:
                    continue  # corrupt row
                spw = sv_won / svpt
                rpw = ret_won / osvpt
                exp_spw = f_avg - (ov["rpw_adj"] - g_avg)
                exp_rpw = g_avg - (ov["spw_adj"] - f_avg)
                vec = np.zeros(len(S_FIELDS))
                vec[SI["w"]] = 1.0
                vec[SI["svpt"]] = svpt
                vec[SI["sv_won"]] = sv_won
                vec[SI["aces"]] = g(me + "ace") or 0.0
                vec[SI["df"]] = g(me + "df") or 0.0
                vec[SI["first_in"]] = g(me + "1stIn") or 0.0
                vec[SI["first_won"]] = fw
                vec[SI["second_won"]] = sw2
                vec[SI["sv_gms"]] = g(me + "SvGms") or 0.0
                vec[SI["bp_faced"]] = g(me + "bpFaced") or 0.0
                vec[SI["bp_saved"]] = g(me + "bpSaved") or 0.0
                vec[SI["ret_pts"]] = osvpt
                vec[SI["ret_won"]] = ret_won
                vec[SI["aces_allowed"]] = g(opp + "ace") or 0.0
                vec[SI["df_induced"]] = g(opp + "df") or 0.0
                ofaced, osaved = g(opp + "bpFaced") or 0.0, g(opp + "bpSaved") or 0.0
                vec[SI["bp_chances"]] = ofaced
                vec[SI["bp_won"]] = max(ofaced - osaved, 0.0)
                vec[SI["ret_gms"]] = g(opp + "SvGms") or 0.0
                vec[SI["spw_resid"]] = spw - exp_spw
                vec[SI["rpw_resid"]] = rpw - exp_rpw
                gw, gl = m.get("winner_games"), m.get("loser_games")
                if status == "completed" and gw is not None and gl is not None and gw == gw and gl == gl:
                    vec[SI["games_won"]] = gw if me == "w_" else gl
                    vec[SI["games_played"]] = gw + gl
                mp.stats.add(t, vec)
                mp.stats_s[surf].add(t, vec)
                mp.stat_matches.append(t)
                if len(mp.stat_matches) > 60:
                    del mp.stat_matches[:-60]
            tot = np.array([w_svpt + l_svpt,
                            (g("w_1stWon") or 0) + (g("w_2ndWon") or 0) + (g("l_1stWon") or 0) + (g("l_2ndWon") or 0),
                            (g("w_ace") or 0) + (g("l_ace") or 0), (g("w_df") or 0) + (g("l_df") or 0)])
            self.tour_avg[surf].add(t, tot)

        # --- rest/fatigue, ranking, age
        mins = g("minutes")
        for st, rk, pt, ag in ((sw, m.get("winner_rank"), m.get("winner_rank_pts"), m.get("winner_age")),
                               (sl, m.get("loser_rank"), m.get("loser_rank_pts"), m.get("loser_age"))):
            st.recent.append((t, mins))
            st.last_t = t
            if rk is not None and rk == rk:
                st.last_rank = float(rk)
            if pt is not None and pt == pt:
                st.last_rank_pts = float(pt)
            if ag is not None and ag == ag:
                st.last_age, st.last_age_t = float(ag), t

        # --- H2H
        key = (w, l) if w < l else (l, w)
        rec = self.h2h.setdefault(key, [0, 0])
        rec[0 if w < l else 1] += 1
        self.last_t = max(self.last_t, t)
        self.matches_seen += 1

    def prune(self, min_last_t: float) -> None:
        """Drop players inactive before ``min_last_t`` (keeps snapshot small)."""
        keep = {p for p, s in self.players.items() if s.last_t is not None and s.last_t >= min_last_t}
        self.players = {p: self.players[p] for p in keep}
        self.h2h = {k: v for k, v in self.h2h.items() if k[0] in keep and k[1] in keep}
