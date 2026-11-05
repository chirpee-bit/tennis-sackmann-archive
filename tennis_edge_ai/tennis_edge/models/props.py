"""Player count props: aces, double faults, break points won.

Each is a Poisson GLM on the log scale with an exposure term from the Markov
model (expected service points for aces/DFs, expected opponent service games
for breaks), fitted as a rate with exposure weights (equivalent to a log-offset
model). Over-dispersion is captured with a negative-binomial dispersion ``alpha``
(Var = mu + alpha * mu^2) chosen by maximum likelihood on the training data.

Each match contributes two player rows (A as the player, then B), which is
symmetric by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor

PROP_MARKETS = ("aces", "double_faults", "break_points_won")

_SPEC = {
    # market: (player-rate feature, opponent-rate feature, tour-rate col, exposure key, target col)
    "aces": ("ace_rate_surf", "ace_allowed_rate", "tour_ace", "svpt_exp", "t_ace"),
    "double_faults": ("df_rate", "df_induced_rate", "tour_df", "svpt_exp", "t_df"),
    "break_points_won": ("bp_won_per_rg", "bp_faced_per_sg", None, "opp_sv_games", "t_bpwon"),
}


def _num(df, c, fill):
    return pd.to_numeric(df[c], errors="coerce").fillna(fill).to_numpy(float) if c in df else np.full(len(df), fill)


def player_rows(df: pd.DataFrame, mk: dict[str, np.ndarray], market: str) -> pd.DataFrame:
    """Stack A-perspective and B-perspective rows with features, exposure and target."""
    prate, orate, tcol, expo, tgt = _SPEC[market]
    rows = []
    tg_half = mk["total_games_mean"] / 2
    for me, op in (("a", "b"), ("b", "a")):
        r = pd.DataFrame({
            "match_id": df["match_id"].to_numpy(), "side": me,
            "player_id": df[f"{me}_id"].to_numpy() if f"{me}_id" in df else None,
            "x_player": np.log(np.clip(_num(df, f"{me}_{prate}", 0.05), 1e-3, None)),
            "x_opp": np.log(np.clip(_num(df, f"{op}_{orate}", 0.05), 1e-3, None)),
            "x_tour": np.log(np.clip(_num(df, tcol, 0.05), 1e-3, None)) if tcol else 0.0,
            "bo5": (_num(df, "best_of", 3) >= 5).astype(float),
            "clay": (df["surface"] == "Clay").to_numpy(float), "grass": (df["surface"] == "Grass").to_numpy(float),
            "x_markov": np.log(np.clip(mk[f"{me}_breaks_exp"] / np.clip(tg_half, 1, None), 1e-3, None)),
            "p_serve": mk["pa"] if me == "a" else mk["pb"],
            "stat_n": _num(df, f"{me}_stat_matches_365", 0),
            "exposure": mk[f"{me}_svpt_exp"] if expo == "svpt_exp" else tg_half,
            "y": pd.to_numeric(df[f"{me}_t_{tgt[2:]}"], errors="coerce").to_numpy(float)
            if f"{me}_t_{tgt[2:]}" in df else np.nan,
        })
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


FEATURES = {
    "aces": ["x_player", "x_opp", "x_tour", "bo5", "clay", "grass", "p_serve"],
    "double_faults": ["x_player", "x_opp", "x_tour", "bo5", "clay", "grass", "p_serve"],
    "break_points_won": ["x_player", "x_opp", "x_markov", "bo5", "clay", "grass"],
}


def nb_pmf(mu: np.ndarray, alpha: float, kmax: int = 60) -> np.ndarray:
    """Negative-binomial (or Poisson if alpha==0) pmf matrix, rows = observations."""
    k = np.arange(kmax + 1)
    mu = np.clip(np.asarray(mu, float), 1e-6, None)[:, None]
    if alpha <= 1e-9:
        return stats.poisson.pmf(k[None, :], mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    return stats.nbinom.pmf(k[None, :], r, p)


class CountPropModel:
    def __init__(self, market: str, l2: float = 1e-4):
        if market not in _SPEC:
            raise ValueError(market)
        self.market = market
        self.l2 = l2
        self.alpha = 0.0

    def fit(self, rows: pd.DataFrame) -> "CountPropModel":
        d = rows[rows["y"].notna() & (rows["exposure"] > 0)]
        X = d[FEATURES[self.market]].to_numpy()
        rate = d["y"].to_numpy() / d["exposure"].to_numpy()
        self.m = PoissonRegressor(alpha=self.l2, max_iter=1000).fit(X, rate, sample_weight=d["exposure"].to_numpy())
        mu = self._mu(d)
        y = d["y"].to_numpy().astype(int)
        best = (-np.inf, 0.0)
        for a in (0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3, 0.45, 0.6):
            pmf = nb_pmf(mu, a, kmax=max(60, int(y.max()) + 1))
            ll = np.mean(np.log(np.clip(pmf[np.arange(len(y)), y], 1e-12, None)))
            if ll > best[0]:
                best = (ll, a)
        self.alpha = best[1]
        self.n_train = len(d)
        return self

    def _mu(self, rows: pd.DataFrame) -> np.ndarray:
        return self.m.predict(rows[FEATURES[self.market]].to_numpy()) * rows["exposure"].to_numpy()

    def predict(self, rows: pd.DataFrame, kmax: int = 60) -> tuple[np.ndarray, np.ndarray]:
        mu = self._mu(rows)
        return mu, nb_pmf(mu, self.alpha, kmax)


def baseline_mean(rows_train: pd.DataFrame, rows_test: pd.DataFrame) -> np.ndarray:
    """Naive baseline: player's shrunk rate x average exposure of the training set (by best-of)."""
    d = rows_train[rows_train["y"].notna()]
    avg_expo = d.groupby("bo5")["exposure"].mean()
    ratio = (d["y"] / np.exp(d["x_player"])).groupby(d["bo5"]).mean()  # y per unit rate
    return np.exp(rows_test["x_player"].to_numpy()) * rows_test["bo5"].map(ratio).fillna(ratio.mean()).to_numpy()
