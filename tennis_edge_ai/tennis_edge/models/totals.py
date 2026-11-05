"""Game/set-count markets from the Markov model.

Serve-point probabilities follow Barnett & Clarke (2005):

    p_A = f_t + (f_A - f_t) - (g_B - g_t)

where f = opponent-adjusted serve points won, g = return points won, and
f_t/g_t are tour surface averages, with a half-weight surface-specific
adjustment. The *sum* p_A + p_B (which drives how many games are played) comes
from these statistics; the *difference* is then solved so the Markov match-win
probability equals the calibrated winner model's probability.

An optional linear correction of p_A + p_B (fitted on training data only) fixes
systematic level bias in game totals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_edge.models.markov import expected_points_per_game, hold_prob, match_distribution, serve_probs_for_target


def _c(df, col, fill):
    return pd.to_numeric(df[col], errors="coerce").fillna(fill).to_numpy(float) if col in df else np.full(len(df), fill)


def raw_serve_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ft = _c(df, "tour_spw", 0.62)
    gt = 1 - ft
    pa = ft + (_c(df, "a_spw_adj", 0.62) - ft) - (_c(df, "b_rpw_adj", 0.38) - gt) \
        + 0.5 * ((_c(df, "a_spw_surf", 0.62) - _c(df, "a_spw", 0.62)) - (_c(df, "b_rpw_surf", 0.38) - _c(df, "b_rpw", 0.38)))
    pb = ft + (_c(df, "b_spw_adj", 0.62) - ft) - (_c(df, "a_rpw_adj", 0.38) - gt) \
        + 0.5 * ((_c(df, "b_spw_surf", 0.62) - _c(df, "b_spw", 0.62)) - (_c(df, "a_rpw_surf", 0.38) - _c(df, "a_rpw", 0.38)))
    return np.clip(pa, 0.35, 0.92), np.clip(pb, 0.35, 0.92)


GH_NODES = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
GH_WEIGHTS = np.array([1 / 6, 2 / 3, 1 / 6])


class MixtureParams:
    """Random-effects parameters for the Markov model.

    Real matches vary in serve dominance and overall hold rates beyond what
    pre-match estimates capture. Because expected games are concave in serve
    dominance, plugging in a point estimate overstates match length (Jensen).
    We integrate the Markov distribution over

        d ~ N(d_hat, sigma_d^2)    (serve-point dominance pa - pb)
        s ~ N(s_hat + shift, sigma_s^2)  (sum pa + pb, i.e. overall hold level)

    with 3x3 Gauss-Hermite quadrature, re-solving d_hat so the *mixture* still
    reproduces the calibrated match-win probability. ``shift``, ``sigma_d`` and
    ``sigma_s`` are fitted by maximum likelihood of observed total games on
    training data only.
    """

    def __init__(self, shift: float = 0.0, sigma_d: float = 0.0, sigma_s: float = 0.0):
        self.shift, self.sigma_d, self.sigma_s = shift, sigma_d, sigma_s

    def as_dict(self) -> dict[str, float]:
        return {"shift": self.shift, "sigma_d": self.sigma_d, "sigma_s": self.sigma_s}

    def fit(self, df: pd.DataFrame, p_win: np.ndarray, max_rows: int = 1000, seed: int = 7) -> "MixtureParams":
        d = df.assign(_pw=np.asarray(p_win, float))
        d = d[(d["status"] == "completed") & d["total_games"].notna()]
        if len(d) < 500:
            return self
        if len(d) > max_rows:
            d = d.sample(max_rows, random_state=seed)
        y = d["total_games"].to_numpy(int)
        pw = d["_pw"].to_numpy()

        def ll(params):
            out = markov_predict(d, pw, MixtureParams(*params))
            pmf = out["total_games_pmf"]
            return float(np.mean(np.log(np.clip(pmf[np.arange(len(y)), np.clip(y, 0, pmf.shape[1] - 1)], 1e-6, 1))))

        cur = [-0.04, 0.06, 0.03]
        best = (ll(tuple(cur)), tuple(cur))
        # coordinate search: shift, sigma_d, sigma_s, then refine shift
        plan = [(0, np.linspace(-0.16, 0.04, 11)), (1, np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.16])),
                (2, np.array([0.0, 0.03, 0.06, 0.09])), (0, None)]
        for k, grid in plan:
            if grid is None:
                grid = best[1][0] + np.array([-0.015, -0.0075, 0.0075, 0.015])
            for v in grid:
                trial = list(best[1])
                trial[k] = float(v)
                val = ll(tuple(trial))
                if val > best[0] + 1e-6:
                    best = (val, tuple(trial))
        self.shift, self.sigma_d, self.sigma_s = best[1]
        self.loglik_ = best[0]
        return self


def _solve_mixture_dominance(s: np.ndarray, target: np.ndarray, bo: np.ndarray, params: MixtureParams,
                             iters: int = 26) -> np.ndarray:
    """Bisection on d_hat so that the quadrature-mixture match-win probability equals ``target``."""
    from tennis_edge.models.markov import match_win_prob
    lo, hi = np.full(s.shape, -0.6), np.full(s.shape, 0.6)
    target = np.clip(target, 1e-4, 1 - 1e-4)
    nodes = list(_nodes(params))
    for _ in range(iters):
        mid = (lo + hi) / 2
        pm = np.zeros_like(s)
        for zd, zs, w in nodes:
            dd = mid + params.sigma_d * zd
            ss = s + params.sigma_s * zs
            pm += w * match_win_prob(np.clip((ss + dd) / 2, 0.2, 0.98), np.clip((ss - dd) / 2, 0.2, 0.98), bo)
        lo = np.where(pm < target, mid, lo)
        hi = np.where(pm >= target, mid, hi)
    return (lo + hi) / 2


def _nodes(params: MixtureParams):
    zi = GH_NODES if params.sigma_d > 0 else np.array([0.0])
    wi = GH_WEIGHTS if params.sigma_d > 0 else np.array([1.0])
    zj = GH_NODES if params.sigma_s > 0 else np.array([0.0])
    wj = GH_WEIGHTS if params.sigma_s > 0 else np.array([1.0])
    for a, wa in zip(zi, wi):
        for b, wb in zip(zj, wj):
            yield a, b, wa * wb


def markov_predict(df: pd.DataFrame, p_win: np.ndarray, params: MixtureParams | None = None,
                   batch: int = 2000) -> dict[str, np.ndarray]:
    """Distributions for every count market. Row order matches ``df``."""
    params = params or MixtureParams()
    pa_raw, pb_raw = raw_serve_probs(df)
    s = np.clip(pa_raw + pb_raw + params.shift, 1.0, 1.6)
    bo = np.where(_c(df, "best_of", 3) >= 5, 5, 3)
    p_win = np.asarray(p_win, float)
    if params.sigma_d == 0 and params.sigma_s == 0:
        pa0, pb0 = serve_probs_for_target(s, p_win, bo)
        d_hat = pa0 - pb0
    else:
        d_hat = _solve_mixture_dominance(s, p_win, bo, params)
    n = len(df)
    out = {"total_games_pmf": np.zeros((n, 72)), "a_games_pmf": np.zeros((n, 36)),
           "b_games_pmf": np.zeros((n, 36)), "sets_pmf": np.zeros((n, 6)), "first_set_pmf": np.zeros((n, 14)),
           "p_match": np.zeros(n), "p_first_set": np.zeros(n), "a_svpt_exp": np.zeros(n), "b_svpt_exp": np.zeros(n),
           "a_breaks_exp": np.zeros(n), "b_breaks_exp": np.zeros(n)}
    for zd, zs, w in _nodes(params):
        dd = d_hat + params.sigma_d * zd
        ss = s + params.sigma_s * zs
        pa = np.clip((ss + dd) / 2, 0.2, 0.98)
        pb = np.clip((ss - dd) / 2, 0.2, 0.98)
        tg_mean = np.zeros(n)
        for best in (3, 5):
            idx = np.where(bo == best)[0]
            for start in range(0, len(idx), batch):
                ii = idx[start:start + batch]
                md = match_distribution(pa[ii], pb[ii], best)
                tg = md.total_games_pmf
                out["total_games_pmf"][ii, :tg.shape[1]] += w * tg
                out["a_games_pmf"][ii, :md.a_games_pmf.shape[1]] += w * md.a_games_pmf
                out["b_games_pmf"][ii, :md.b_games_pmf.shape[1]] += w * md.b_games_pmf
                out["sets_pmf"][ii] += w * md.sets_pmf
                out["first_set_pmf"][ii] += w * md.first_set_pmf
                out["p_match"][ii] += w * md.p_match
                out["p_first_set"][ii] += w * md.p_first_set
                tg_mean[ii] = tg @ np.arange(tg.shape[1])
        # expected service games/points and breaks (inputs to ace / DF / break-point models)
        sg = tg_mean / 2
        out["a_svpt_exp"] += w * sg * expected_points_per_game(pa)
        out["b_svpt_exp"] += w * sg * expected_points_per_game(pb)
        out["a_breaks_exp"] += w * sg * (1 - hold_prob(pb))
        out["b_breaks_exp"] += w * sg * (1 - hold_prob(pa))
    out["pa"], out["pb"] = np.clip((s + d_hat) / 2, 0.2, 0.98), np.clip((s - d_hat) / 2, 0.2, 0.98)
    k72, k36 = np.arange(72), np.arange(36)
    out["total_games_mean"] = out["total_games_pmf"] @ k72
    out["a_games_mean"] = out["a_games_pmf"] @ k36
    out["b_games_mean"] = out["b_games_pmf"] @ k36
    out["sets_mean"] = out["sets_pmf"] @ np.arange(6)
    out["first_set_mean"] = out["first_set_pmf"] @ np.arange(14)
    return out
