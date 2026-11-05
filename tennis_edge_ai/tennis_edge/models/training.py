"""End-to-end walk-forward evaluation and production model fitting.

``run_tour`` produces, for one tour:

1. Out-of-sample (OOS) match-winner predictions for every test year and every
   candidate model (baselines, logistic, LightGBM; raw and calibrated).
2. The production winner model, chosen by pooled OOS log loss. LightGBM is only
   chosen if it beats the best logistic variant with a paired z < -2.
3. OOS predictions for count markets (total games, first-set games, total sets,
   player games won, aces, double faults, break points won) from the
   random-effects Markov model + count GLMs, next to naive baselines.
4. Isotonic line-calibrators per market, kept only if they improve pooled OOS
   Brier score in a nested walk-forward.
5. Production artifacts fitted on the most recent data, registered in
   ``model_versions`` with their validation metrics.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from tennis_edge.config import get_settings
from tennis_edge.db.database import Database, utcnow_iso
from tennis_edge.features.builder import load_features
from tennis_edge.models.markov import prob_over
from tennis_edge.models.props import PROP_MARKETS, CountPropModel, baseline_mean, player_rows
from tennis_edge.models.totals import MixtureParams, markov_predict
from tennis_edge.models.winner import MODEL_FACTORIES, TemperatureCalibrator
from tennis_edge.validation.metrics import binary_summary, numeric_summary
from tennis_edge.validation.walk_forward import diebold_mariano_logloss, eligible_winner_rows, walk_forward_winner

log = logging.getLogger(__name__)

STORE_WINNER_VARIANTS = ("elo_baseline_cal", "rank_baseline_cal", "logistic", "logistic_cal", "lightgbm",
                         "lightgbm_cal")
MATCH_COUNT_MARKETS = ("total_games", "first_set_games", "total_sets")
ALL_COUNT_MARKETS = MATCH_COUNT_MARKETS + ("player_games_won",) + PROP_MARKETS


@dataclass
class TourModels:
    """Production bundle for one tour (pickled with joblib)."""
    tour: str
    version: str
    winner_name: str
    winner_model: object
    temperature: TemperatureCalibrator
    mixture: MixtureParams
    props: dict[str, CountPropModel]
    line_calibrators: dict[str, IsotonicRegression] = field(default_factory=dict)
    market_metrics: dict[str, dict] = field(default_factory=dict)
    trained_through: str = ""
    created_at: str = ""

    def winner_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.winner_model.predict_proba(df)
        return self.temperature.transform(p) if self.winner_name.endswith("_cal") else p

    def calibrate_line(self, market: str, p_over: np.ndarray) -> np.ndarray:
        cal = self.line_calibrators.get(market)
        return np.clip(cal.predict(p_over), 0.001, 0.999) if cal is not None else p_over


def _base_name(variant: str) -> str:
    return variant[:-4] if variant.endswith("_cal") else variant


def choose_winner_model(wf: pd.DataFrame) -> tuple[str, dict]:
    pooled = {m: binary_summary(g["y"], g["prob"]) for m, g in wf.groupby("model")}
    logistic = min((m for m in pooled if m.startswith("logistic")), key=lambda m: pooled[m]["log_loss"])
    choice, evidence = logistic, {}
    gbm = [m for m in pooled if m.startswith("lightgbm")]
    if gbm:
        best_gbm = min(gbm, key=lambda m: pooled[m]["log_loss"])
        a = wf[wf.model == best_gbm].set_index("match_id")
        b = wf[wf.model == logistic].set_index("match_id").loc[a.index]
        dm = diebold_mariano_logloss(a["y"].to_numpy(), a["prob"].to_numpy(), b["prob"].to_numpy())
        evidence = {"lightgbm_variant": best_gbm, "vs": logistic, **dm}
        if dm["z"] < -2.0:
            choice = best_gbm
    return choice, {"pooled": pooled, "gbm_vs_logistic": evidence}


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["status"] == "completed") & df["total_games"].notna()]


def _line_for(mean: np.ndarray) -> np.ndarray:
    return np.floor(mean) + 0.5


def count_market_records(test: pd.DataFrame, mk: dict, prop_preds: dict, base: dict) -> list[pd.DataFrame]:
    """Long-format OOS records for every count market (model + baseline)."""
    out = []
    meta = {"match_id": test["match_id"].to_numpy(), "match_date": pd.to_datetime(test["date"]).dt.date.astype(str).to_numpy(),
            "surface": test["surface"].to_numpy(), "level": test["level_group"].to_numpy(),
            "best_of": test["best_of"].to_numpy()}

    def emit(market, model_name, pred_mean, actual, pmf=None, line=None, prob=None, ids=None):
        actual = np.asarray(actual, float)
        line = _line_for(base[market]) if line is None else line
        if prob is None:
            prob, _, _ = prob_over(pmf, line)
        d = pd.DataFrame({**({k: np.tile(v, len(actual) // len(v)) for k, v in meta.items()}),
                          "market": market, "model_name": model_name, "line": line, "prob": prob,
                          "outcome": (actual > line).astype(float), "pred_mean": pred_mean, "actual": actual})
        if ids is not None:
            d["match_id"] = ids
        out.append(d[~np.isnan(actual)])

    emit("total_games", "markov_mixture", mk["total_games_mean"], test["total_games"], mk["total_games_pmf"])
    emit("first_set_games", "markov_mixture", mk["first_set_mean"], test["first_set_games"], mk["first_set_pmf"],
         line=np.full(len(test), 9.5))
    emit("total_sets", "markov_mixture", mk["sets_mean"], test["total_sets"], mk["sets_pmf"],
         line=np.where(test["best_of"].to_numpy() >= 5, 3.5, 2.5))
    pg_actual = np.concatenate([test["a_games"].to_numpy(float), test["b_games"].to_numpy(float)])
    pg_pmf = np.vstack([mk["a_games_pmf"], mk["b_games_pmf"]])
    pg_mean = np.concatenate([mk["a_games_mean"], mk["b_games_mean"]])
    ids = np.concatenate([test["match_id"].to_numpy() + ":A", test["match_id"].to_numpy() + ":B"])
    emit("player_games_won", "markov_mixture", pg_mean, pg_actual, pg_pmf, ids=ids)
    for market, (mu, pmf, actual) in prop_preds.items():
        emit(market, "count_glm", mu, actual, pmf, ids=ids)
    # baselines: constant mean, P(over) = training frequency over that line
    for market, b in base.items():
        if market.startswith("__"):
            continue
        actual = {"total_games": test["total_games"], "first_set_games": test["first_set_games"],
                  "total_sets": test["total_sets"], "player_games_won": pg_actual}.get(market)
        if actual is None:
            actual = prop_preds[market][2]
        line = {"first_set_games": np.full(len(test), 9.5),
                "total_sets": np.where(test["best_of"].to_numpy() >= 5, 3.5, 2.5)}.get(market)
        line = _line_for(b) if line is None else line
        prob = base[f"__p_{market}"] if f"__p_{market}" in base else np.full(len(b), np.nan)
        emit(market, "baseline", b, actual, line=line, prob=prob,
             ids=ids if len(b) == 2 * len(test) else None)
    return out


def _baselines(train: pd.DataFrame, test: pd.DataFrame, prop_rows_train: dict, prop_rows_test: dict) -> dict:
    tr = _completed(train)
    base: dict[str, np.ndarray] = {}
    key_te = list(zip(test["best_of"], test["surface"]))

    def grouped_mean(col):
        g = tr.groupby(["best_of", "surface"])[col].mean()
        overall = tr.groupby("best_of")[col].mean()
        return np.array([g.get(k, overall.get(k[0], tr[col].mean())) for k in key_te], dtype=float)

    def freq_over(col, line_vals):
        vals = tr[[col, "best_of"]].dropna()
        return np.array([(vals[vals.best_of == bo][col] > ln).mean() for bo, ln in zip(test["best_of"], line_vals)])

    for m, col in (("total_games", "total_games"), ("first_set_games", "first_set_games"), ("total_sets", "total_sets")):
        base[m] = grouped_mean(col)
        lines = {"first_set_games": np.full(len(test), 9.5),
                 "total_sets": np.where(test["best_of"].to_numpy() >= 5, 3.5, 2.5)}.get(m, _line_for(base[m]))
        base[f"__p_{m}"] = freq_over(col, lines)
    # player games won: training mean by best_of, games share
    games_tr = pd.concat([tr[["best_of", "a_games"]].rename(columns={"a_games": "g"}),
                          tr[["best_of", "b_games"]].rename(columns={"b_games": "g"})])
    gm = games_tr.groupby("best_of")["g"].mean()
    share_a = pd.to_numeric(test["a_games_share"], errors="coerce").fillna(0.5).to_numpy()
    share_b = pd.to_numeric(test["b_games_share"], errors="coerce").fillna(0.5).to_numpy()
    tot = base["total_games"]
    base["player_games_won"] = np.concatenate([tot * share_a, tot * share_b])
    pg_line = _line_for(base["player_games_won"])
    bo2 = np.concatenate([test["best_of"].to_numpy(), test["best_of"].to_numpy()])
    # player-level baselines get a Poisson distribution around their own mean (fair comparison)
    from scipy import stats as _st
    base["__p_player_games_won"] = 1 - _st.poisson.cdf(np.floor(pg_line), np.clip(base["player_games_won"], 1e-6, None))
    del gm
    for market in PROP_MARKETS:
        b = baseline_mean(prop_rows_train[market], prop_rows_test[market])
        base[market] = b
        base[f"__p_{market}"] = 1 - _st.poisson.cdf(np.floor(_line_for(b)), np.clip(b, 1e-6, None))
    return base


def run_tour(db: Database, tour: str, first_test_year: int | None = None, window: int | None = None,
             include_lgbm: bool = True, progress: Callable[[str], None] | None = None,
             store: bool = True) -> dict:
    s = get_settings()
    first_test_year = first_test_year or s.first_test_year
    window = window or s.train_window_years
    say = progress or (lambda m: log.info(m))
    feats = load_features(tour)
    if feats is None:
        raise RuntimeError(f"features for {tour} not built")
    run_id = f"{tour}-{utcnow_iso()}-{uuid.uuid4().hex[:6]}"
    models = ("elo_baseline", "rank_baseline", "logistic") + (("lightgbm",) if include_lgbm else ())
    say(f"[{tour}] winner walk-forward: models={models}")
    wf = walk_forward_winner(feats, first_test_year, window, models, progress=say)
    choice, evidence = choose_winner_model(wf)
    say(f"[{tour}] chosen winner model: {choice}")

    el = eligible_winner_rows(feats)
    years = pd.to_datetime(el["date"]).dt.year
    chosen_oos = wf[wf.model == choice].set_index("match_id")["prob"]
    records: list[pd.DataFrame] = []
    mixture_by_year = {}
    for Y in sorted(wf["year"].unique()):
        test = el[(years == Y) & el["match_id"].isin(chosen_oos.index)]
        test = test[test["total_games"].notna() | test["a_t_ace"].notna()]
        if test.empty:
            continue
        train = el[(years >= Y - window) & (years < Y)]
        recent = train[pd.to_datetime(train["date"]).dt.year >= Y - 3]
        base_model = MODEL_FACTORIES[_base_name(choice)]().fit(train, train["y"].to_numpy())
        cal = TemperatureCalibrator()
        if choice.endswith("_cal"):
            cal.T = float(wf[(wf.model == choice) & (wf.year == Y)]["temperature"].iloc[0])
        p_recent = cal.transform(base_model.predict_proba(recent))
        import time as _time
        t0 = _time.time()
        mix = MixtureParams().fit(recent, p_recent)
        mixture_by_year[int(Y)] = mix.as_dict()
        say(f"[{tour}] {Y}: mixture fitted in {_time.time() - t0:.0f}s")
        p_test = chosen_oos.loc[test["match_id"]].to_numpy()
        mk_test = markov_predict(test, p_test, mix)
        sub = recent.sample(min(len(recent), 8000), random_state=s.random_seed)
        mk_sub = markov_predict(sub, cal.transform(base_model.predict_proba(sub)), mix)
        prop_preds, rows_tr, rows_te = {}, {}, {}
        for market in PROP_MARKETS:
            rows_tr[market] = player_rows(sub, mk_sub, market)
            rows_te[market] = player_rows(test, mk_test, market)
            pm = CountPropModel(market).fit(rows_tr[market])
            mu, pmf = pm.predict(rows_te[market])
            prop_preds[market] = (mu, pmf, rows_te[market]["y"].to_numpy())
        base = _baselines(train[pd.to_datetime(train["date"]).dt.year >= Y - 3], test, rows_tr, rows_te)
        recs = count_market_records(test, mk_test, prop_preds, base)
        records.extend(r.assign(year=int(Y)) for r in recs)
        say(f"[{tour}] {Y}: counts done (n={len(test)}, mixture={mix.as_dict()})")
    counts = pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    # ---------------- nested isotonic calibration check per market
    calib_report, keep_cal = {}, {}
    if not counts.empty:
        counts["prob_cal"] = np.nan
        for market in ALL_COUNT_MARKETS:
            mm = counts[(counts.market == market) & (counts.model_name != "baseline")]
            for Y in sorted(mm["year"].unique()):
                prior = mm[mm.year < Y]
                if len(prior) < 2000:
                    continue
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(prior["prob"], prior["outcome"])
                idx = mm.index[mm.year == Y]
                counts.loc[idx, "prob_cal"] = iso.predict(counts.loc[idx, "prob"])
            ev = counts[(counts.market == market) & (counts.model_name != "baseline") & counts.prob_cal.notna()]
            if len(ev):
                raw = binary_summary(ev["outcome"], ev["prob"])
                cald = binary_summary(ev["outcome"], ev["prob_cal"])
                calib_report[market] = {"raw_brier": raw["brier"], "cal_brier": cald["brier"],
                                        "raw_ece": raw["ece"], "cal_ece": cald["ece"], "n": raw["n"]}
                keep_cal[market] = cald["brier"] < raw["brier"] - 1e-4
    say(f"[{tour}] line calibration kept: {keep_cal}")

    # ---------------- metrics per market
    market_metrics: dict[str, dict] = {"match_winner": evidence["pooled"].get(choice, {}) | {"model": choice}}
    for market in ALL_COUNT_MARKETS:
        d = counts[counts.market == market] if not counts.empty else pd.DataFrame()
        if d.empty:
            continue
        mdl = d[d.model_name != "baseline"]
        bl = d[d.model_name == "baseline"]
        use_cal = keep_cal.get(market, False)
        pcol = mdl["prob_cal"].where(mdl["prob_cal"].notna(), mdl["prob"]) if use_cal else mdl["prob"]
        market_metrics[market] = {
            "model": mdl["model_name"].iloc[0], "line_calibrated": bool(use_cal),
            **{f"model_{k}": v for k, v in numeric_summary(mdl["actual"], mdl["pred_mean"]).items()},
            **{f"base_{k}": v for k, v in numeric_summary(bl["actual"], bl["pred_mean"]).items()},
            **{f"model_line_{k}": v for k, v in binary_summary(mdl["outcome"], pcol).items()},
            **{f"base_line_{k}": v for k, v in binary_summary(bl["outcome"], bl["prob"].fillna(0.5)).items()},
            "beats_baseline_mae": bool(numeric_summary(mdl["actual"], mdl["pred_mean"])["mae"] <
                                       numeric_summary(bl["actual"], bl["pred_mean"])["mae"]),
            "beats_baseline_brier": bool(binary_summary(mdl["outcome"], pcol)["brier"] <
                                         binary_summary(bl["outcome"], bl["prob"].fillna(0.5))["brier"]),
        }

    # ---------------- store OOS predictions
    if store:
        _store_backtests(db, run_id, tour, wf, counts, keep_cal)

    # ---------------- production fit
    say(f"[{tour}] fitting production models")
    bundle = fit_production(feats, tour, choice, wf, window, keep_cal, counts, market_metrics)
    path = s.artifacts_dir / f"models_{tour}.joblib"
    joblib.dump(bundle, path)
    params = {"winner_model": choice, "window_years": window, "first_test_year": first_test_year,
              "mixture": bundle.mixture.as_dict(), "temperature": bundle.temperature.T,
              "prop_alpha": {k: v.alpha for k, v in bundle.props.items()},
              "line_calibrated": keep_cal, "mixture_by_year": mixture_by_year}
    if store:
        db.execute("UPDATE model_versions SET is_active=0 WHERE tour=?", (tour,))
        db.execute("INSERT INTO model_versions(version,created_at,tour,market,model_type,params_json,train_start,"
                   "train_end,metrics_json,artifact_path,is_active) VALUES (?,?,?,?,?,?,?,?,?,?,1)",
                   (bundle.version, bundle.created_at, tour, "all", choice, json.dumps(params, default=float),
                    str(int(pd.to_datetime(feats["date"]).dt.year.max()) - window), bundle.trained_through,
                    json.dumps({"markets": market_metrics, "winner_candidates": evidence["pooled"],
                                "gbm_vs_logistic": evidence["gbm_vs_logistic"], "line_calibration": calib_report,
                                "run_id": run_id}, default=float), str(path)))
        db.audit("train_models", "model_versions", bundle.version, {"tour": tour, "run_id": run_id})
    return {"run_id": run_id, "version": bundle.version, "choice": choice, "metrics": market_metrics,
            "calibration": calib_report, "evidence": evidence}


def _store_backtests(db: Database, run_id: str, tour: str, wf: pd.DataFrame, counts: pd.DataFrame,
                     keep_cal: dict) -> None:
    rows = []
    w = wf[wf.model.isin(STORE_WINNER_VARIANTS)]
    for r in w.itertuples(index=False):
        rows.append((run_id, r.model, tour, "match_winner", r.match_id, str(pd.Timestamp(r.date).date()), r.surface,
                     r.level_group, int(r.best_of), None, float(r.prob), float(r.y), None, None))
    if not counts.empty:
        for r in counts.itertuples(index=False):
            prob = r.prob_cal if (keep_cal.get(r.market) and r.model_name != "baseline" and r.prob_cal == r.prob_cal) else r.prob
            rows.append((run_id, r.model_name, tour, r.market, r.match_id, r.match_date, r.surface, r.level,
                         int(r.best_of), float(r.line), None if prob != prob else float(prob), float(r.outcome),
                         float(r.pred_mean), float(r.actual)))
    db.executemany("INSERT OR IGNORE INTO backtest_predictions(run_id,model_name,tour,market,match_id,match_date,"
                   "surface,level,best_of,line,prob,outcome,pred_mean,actual) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   rows)
    db.set_meta(f"latest_run_{tour}", run_id)


def fit_production(feats: pd.DataFrame, tour: str, choice: str, wf: pd.DataFrame, window: int,
                   keep_cal: dict, counts: pd.DataFrame, market_metrics: dict) -> TourModels:
    s = get_settings()
    el = eligible_winner_rows(feats)
    years = pd.to_datetime(el["date"]).dt.year
    last = int(years.max())
    train = el[years >= last - window + 1]
    model = MODEL_FACTORIES[_base_name(choice)]().fit(train, train["y"].to_numpy())
    # temperature from the most recent full OOS year (model trained before it)
    temp = TemperatureCalibrator()
    if choice.endswith("_cal"):
        recent_years = sorted(wf["year"].unique())
        ref_year = recent_years[-2] if len(recent_years) > 1 else recent_years[-1]
        g = wf[(wf.model == _base_name(choice)) & (wf.year >= ref_year)]
        temp.fit(g["prob"].to_numpy(), g["y"].to_numpy())
    recent = train[years.loc[train.index] >= last - 2]
    p_recent = temp.transform(model.predict_proba(recent)) if choice.endswith("_cal") else model.predict_proba(recent)
    mix = MixtureParams().fit(recent, p_recent)
    sub = recent.sample(min(len(recent), 15000), random_state=s.random_seed)
    p_sub = temp.transform(model.predict_proba(sub)) if choice.endswith("_cal") else model.predict_proba(sub)
    mk = markov_predict(sub, p_sub, mix)
    props = {m: CountPropModel(m).fit(player_rows(sub, mk, m)) for m in PROP_MARKETS}
    cals = {}
    for market, keep in keep_cal.items():
        if keep:
            mm = counts[(counts.market == market) & (counts.model_name != "baseline")]
            cals[market] = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(mm["prob"], mm["outcome"])
    created = utcnow_iso()
    version = f"{tour}-v{pd.Timestamp(created).strftime('%Y%m%d%H%M%S')}-{choice}"
    return TourModels(tour=tour, version=version, winner_name=choice, winner_model=model, temperature=temp,
                      mixture=mix, props=props, line_calibrators=cals, market_metrics=market_metrics,
                      trained_through=str(pd.to_datetime(feats["date"]).max().date()), created_at=created)


def load_tour_models(tour: str) -> TourModels | None:
    path = get_settings().artifacts_dir / f"models_{tour}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
