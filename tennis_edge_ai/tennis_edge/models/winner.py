"""Match-winner models.

* ``EloBaseline`` - surface-blended Elo probability, no fitting.
* ``RankBaseline`` - logistic regression on log-rank difference.
* ``LogisticWinner`` - symmetric logistic regression on A-minus-B features.
* ``LGBMWinner`` - gradient boosting on the same features (+ context); kept only
  if walk-forward validation shows it beats the logistic model.

Symmetry: training rows are augmented with their mirror image (features negated,
label flipped) and the logistic model has no intercept, so P(A beats B) =
1 - P(B beats A) exactly. LightGBM predictions are symmetrised by averaging
p(x) and 1 - p(mirror(x)).

Calibration: temperature scaling (logit slope, no intercept) fitted on
out-of-sample predictions from the preceding period.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DIFF_FEATURES = ["elo_blend_d", "elo_surf_d", "elo_d", "lrank_d", "lpts_d", "sr_dom_d", "spw_surf_d",
                 "rpw_surf_d", "form_d", "win_rate_d", "surf_win_rate_d", "h2h_d", "rest_d", "m14_d",
                 "min14_d", "age_d", "age2_d", "ht_d", "exp_d", "elo_bo5", "sr_bo5"]
CONTEXT_FEATURES = ["bo5", "surf_code", "lvl_code"]

_SURF = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
_LVL = {"Grand Slam": 0, "Masters 1000": 1, "WTA 1000": 1, "ATP 250/500": 2, "WTA 500": 2, "WTA 250/500": 2,
        "Tour Finals": 3, "Challenger": 4, "WTA 125": 4, "Olympics": 5, "Team (Davis Cup)": 6,
        "Team (BJK Cup)": 6, "Futures/ITF": 7, "Other": 8}


def _col(df: pd.DataFrame, c: str, fill: float) -> np.ndarray:
    if c not in df:
        return np.full(len(df), fill)
    return pd.to_numeric(df[c], errors="coerce").fillna(fill).to_numpy(dtype=float)


def design(df: pd.DataFrame) -> pd.DataFrame:
    """A-minus-B feature matrix from oriented feature rows (pre-match only)."""
    tour_ht = 185.0 if (df["tour"].iloc[0] if len(df) else "ATP") == "ATP" else 172.0
    X = pd.DataFrame(index=df.index)
    X["elo_blend_d"] = (_col(df, "a_elo_blend", 1500) - _col(df, "b_elo_blend", 1500)) / 400
    X["elo_surf_d"] = (_col(df, "a_elo_surf", 1500) - _col(df, "b_elo_surf", 1500)) / 400
    X["elo_d"] = (_col(df, "a_elo", 1500) - _col(df, "b_elo", 1500)) / 400
    X["lrank_d"] = np.log(_col(df, "a_rank", 1500)) - np.log(_col(df, "b_rank", 1500))
    X["lpts_d"] = np.log1p(_col(df, "a_rank_pts", 0)) - np.log1p(_col(df, "b_rank_pts", 0))
    a_sr = _col(df, "a_spw_adj", 0.62) + _col(df, "a_rpw_adj", 0.38)
    b_sr = _col(df, "b_spw_adj", 0.62) + _col(df, "b_rpw_adj", 0.38)
    X["sr_dom_d"] = (a_sr - b_sr) * 10
    X["spw_surf_d"] = (_col(df, "a_spw_surf", 0.62) - _col(df, "b_spw_surf", 0.62)) * 10
    X["rpw_surf_d"] = (_col(df, "a_rpw_surf", 0.38) - _col(df, "b_rpw_surf", 0.38)) * 10
    X["form_d"] = (_col(df, "a_form", 0) - _col(df, "b_form", 0)) * 5
    X["win_rate_d"] = _col(df, "a_win_rate", .5) - _col(df, "b_win_rate", .5)
    X["surf_win_rate_d"] = _col(df, "a_surf_win_rate", .5) - _col(df, "b_surf_win_rate", .5)
    ha, hb = _col(df, "h2h_a", 0), _col(df, "h2h_b", 0)
    X["h2h_d"] = (ha - hb) / (ha + hb + 2)
    X["rest_d"] = (np.clip(_col(df, "a_days_rest", 14), 0, 30) - np.clip(_col(df, "b_days_rest", 14), 0, 30)) / 10
    X["m14_d"] = (_col(df, "a_m14", 0) - _col(df, "b_m14", 0)) / 3
    X["min14_d"] = (_col(df, "a_min14", 0) - _col(df, "b_min14", 0)) / 300
    aa, ab = _col(df, "a_age", 26), _col(df, "b_age", 26)
    X["age_d"] = (aa - ab) / 5
    X["age2_d"] = ((aa - 26) ** 2 - (ab - 26) ** 2) / 50
    X["ht_d"] = (_col(df, "a_ht", tour_ht) - _col(df, "b_ht", tour_ht)) / 10
    X["exp_d"] = np.log1p(_col(df, "a_elo_n", 0)) - np.log1p(_col(df, "b_elo_n", 0))
    bo5 = (_col(df, "best_of", 3) >= 5).astype(float)
    X["elo_bo5"] = X["elo_blend_d"] * bo5
    X["sr_bo5"] = X["sr_dom_d"] * bo5
    X["bo5"] = bo5
    X["surf_code"] = df["surface"].map(_SURF).fillna(0).to_numpy() if "surface" in df else 0
    X["lvl_code"] = df["level_group"].map(_LVL).fillna(8).to_numpy() if "level_group" in df else 8
    return X


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    M = X.copy()
    M[DIFF_FEATURES] = -M[DIFF_FEATURES]
    return M


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class EloBaseline:
    name = "elo_baseline"

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "EloBaseline":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        d = _col(df, "a_elo_blend", 1500) - _col(df, "b_elo_blend", 1500)
        return 1 / (1 + 10 ** (-d / 400))


class RankBaseline:
    name = "rank_baseline"

    def fit(self, df, y):
        x = design(df)[["lrank_d"]].to_numpy()
        self.m = LogisticRegression(fit_intercept=False, C=1.0).fit(np.vstack([x, -x]), np.concatenate([y, 1 - y]))
        return self

    def predict_proba(self, df):
        return self.m.predict_proba(design(df)[["lrank_d"]].to_numpy())[:, 1]


class LogisticWinner:
    name = "logistic"

    def __init__(self, C: float = 0.5):
        self.C = C

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "LogisticWinner":
        X = design(df)[DIFF_FEATURES].to_numpy()
        self.scale_ = np.sqrt((X ** 2).mean(axis=0)) + 1e-9      # no centring: keeps antisymmetry
        Z = X / self.scale_
        self.m = LogisticRegression(fit_intercept=False, C=self.C, max_iter=2000)
        self.m.fit(np.vstack([Z, -Z]), np.concatenate([y, 1 - y]))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        Z = design(df)[DIFF_FEATURES].to_numpy() / self.scale_
        return self.m.predict_proba(Z)[:, 1]

    def coefficients(self) -> dict[str, float]:
        return dict(zip(DIFF_FEATURES, (self.m.coef_[0] / self.scale_).round(4)))


class LGBMWinner:
    name = "lightgbm"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.03, num_leaves: int = 15,
                 min_child_samples: int = 200, seed: int = 7):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1,
                           n_jobs=int(os.environ.get("TE_LGBM_THREADS", "2")))

    def fit(self, df, y):
        import lightgbm as lgb
        X = design(df)
        cols = DIFF_FEATURES + CONTEXT_FEATURES
        XX = pd.concat([X[cols], mirror(X)[cols]], ignore_index=True)
        self.m = lgb.LGBMClassifier(**self.params).fit(XX, np.concatenate([y, 1 - y]))
        self.cols = cols
        return self

    def predict_proba(self, df):
        X = design(df)
        p = self.m.predict_proba(X[self.cols])[:, 1]
        pm = self.m.predict_proba(mirror(X)[self.cols])[:, 1]
        return 0.5 * (p + 1 - pm)


class TemperatureCalibrator:
    """p' = sigmoid(T * logit(p)). T fitted by maximum likelihood (keeps symmetry)."""

    def __init__(self):
        self.T = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        z = _logit(np.asarray(p))[:, None]
        if len(z) >= 200:
            m = LogisticRegression(fit_intercept=False, C=1e4).fit(np.vstack([z, -z]),
                                                                    np.concatenate([y, 1 - y]))
            self.T = float(m.coef_[0][0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.T * _logit(np.asarray(p))))


MODEL_FACTORIES = {"elo_baseline": EloBaseline, "rank_baseline": RankBaseline,
                   "logistic": LogisticWinner, "lightgbm": LGBMWinner}
