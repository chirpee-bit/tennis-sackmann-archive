"""Page 5: Model performance - walk-forward out-of-sample evidence and live tracked results."""
import json

import ui  # noqa: F401
from ui import SERIES, MUTED, footer, get_database, num, pct, setup_page, style_fig

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tennis_edge.analytics.performance import (backtest_calibration, backtest_frame, backtest_summary,
                                               simulated_flat_roi, tracked_breakdown)
from tennis_edge.validation.drift import performance_drift

setup_page("Model Performance", "All numbers below are out-of-sample: chronological walk-forward, never random splits")
db = get_database()

versions = db.df("SELECT version, created_at, tour, model_type, train_end, is_active, metrics_json, params_json "
                 "FROM model_versions ORDER BY created_at DESC")
if versions.empty:
    st.warning("No trained models yet. Run `python scripts/train_models.py`.")
    footer()
    st.stop()

tour = st.radio("Tour", sorted(versions["tour"].unique()), horizontal=True)
vt = versions[versions["tour"] == tour]
chosen = st.selectbox("Model version", vt["version"], index=0)
v = vt[vt["version"] == chosen].iloc[0]
metrics = json.loads(v["metrics_json"])
run_id = metrics.get("run_id")
params = json.loads(v["params_json"])


@st.cache_data(ttl=600, show_spinner=False)
def load_bt(run: str, market: str) -> pd.DataFrame:
    return backtest_frame(get_database(), [run], market)


tab_w, tab_c, tab_live, tab_ver = st.tabs(["Match winner", "Count markets", "Live tracked results", "Model versions"])

with tab_w:
    bt = load_bt(run_id, "match_winner")
    if bt.empty:
        st.info("No stored backtest predictions for this run.")
    else:
        st.markdown(f"Production model: **{v['model_type']}** · test years from {params.get('first_test_year')} · "
                    f"{bt['match_id'].nunique():,} out-of-sample matches")
        comp = backtest_summary(bt, ["model_name"]).sort_values("log_loss")
        comp_view = comp[["model_name", "n", "accuracy", "brier", "log_loss", "ece"]].copy()
        st.dataframe(comp_view.style.format({"accuracy": "{:.2%}", "brier": "{:.4f}", "log_loss": "{:.4f}",
                                             "ece": "{:.4f}", "n": "{:,}"}), hide_index=True, width="stretch")
        ev = metrics.get("gbm_vs_logistic", {})
        if ev:
            verdict = "kept" if v["model_type"].startswith("lightgbm") else "NOT used"
            st.caption(f"Gradient boosting {verdict}: paired log-loss difference {ev.get('mean_diff', 0):+.5f} "
                       f"(z = {ev.get('z', 0):.2f}; it must beat logistic with z < -2).")
        models_avail = list(comp["model_name"])
        sel = st.multiselect("Calibration chart models", models_avail,
                             default=[m for m in (v["model_type"], "elo_baseline_cal") if m in models_avail])
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                                     name="perfect", hoverinfo="skip"))
            for i, m in enumerate(sel):
                cal = backtest_calibration(bt[bt.model_name == m], 10)
                fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["observed"], mode="lines+markers", name=m,
                                         line=dict(color=SERIES[i % len(SERIES)], width=2), marker=dict(size=8),
                                         customdata=cal["n"],
                                         hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<br>n=%{customdata}<extra>" + m + "</extra>"))
            fig.update_layout(title="Reliability diagram", xaxis_title="Predicted probability",
                              yaxis_title="Observed frequency", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig, 380), width="stretch")
        with c2:
            prod = bt[bt.model_name == v["model_type"]].copy()
            prod["month"] = pd.to_datetime(prod["match_date"]).dt.to_period("Q").astype(str)
            mon = backtest_summary(prod, ["month"])
            fig = go.Figure(go.Scatter(x=mon["month"], y=mon["log_loss"], mode="lines+markers", name="log loss",
                                       line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                                       hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
            fig.update_layout(title=f"Quarterly log loss · {v['model_type']}", yaxis_title="log loss (lower is better)")
            st.plotly_chart(style_fig(fig, 380, legend=False), width="stretch")
        prod = bt[bt.model_name == v["model_type"]]
        c1, c2, c3 = st.columns(3)
        for col, by, label in ((c1, "surface", "By surface"), (c2, "level", "By tournament level"),
                               (c3, "best_of", "By format")):
            with col:
                st.markdown(f"**{label}**")
                t = backtest_summary(prod, [by])[[by, "n", "accuracy", "brier", "log_loss", "ece"]]
                st.dataframe(t.style.format({"accuracy": "{:.1%}", "brier": "{:.4f}", "log_loss": "{:.4f}",
                                             "ece": "{:.3f}"}), hide_index=True, width="stretch")
        d = performance_drift(prod)
        msg = {"stable": "ok", "watch": "warn", "DRIFT": "bad"}.get(d.get("status"), "warn")
        st.markdown(f"{ui.badge('DRIFT CHECK', msg)} {d.get('status')} — recent 90-day Brier "
                    f"{num(d.get('brier_recent'), 4)} vs reference {num(d.get('brier_reference'), 4)} "
                    f"(z = {num(d.get('z'), 2)}, n = {d.get('n_recent', 0)})", unsafe_allow_html=True)

with tab_c:
    mm = metrics.get("markets", {})
    rows = []
    for market, m in mm.items():
        if market == "match_winner":
            continue
        rows.append({"market": market, "model": m.get("model"), "n": m.get("model_n"),
                     "MAE model": m.get("model_mae"), "MAE baseline": m.get("base_mae"),
                     "RMSE model": m.get("model_rmse"), "RMSE baseline": m.get("base_rmse"),
                     "Brier@line model": m.get("model_line_brier"), "Brier@line baseline": m.get("base_line_brier"),
                     "ECE model": m.get("model_line_ece"), "line-calibrated": m.get("line_calibrated"),
                     "beats baseline (Brier)": m.get("beats_baseline_brier"), "beats baseline (MAE)": m.get("beats_baseline_mae")})
    if rows:
        t = pd.DataFrame(rows)
        st.dataframe(t.style.format({c: "{:.4f}" for c in t.columns if "Brier" in c or "ECE" in c} |
                                    {c: "{:.3f}" for c in t.columns if "MAE" in c or "RMSE" in c} | {"n": "{:,}"}),
                     hide_index=True, width="stretch")
        st.caption("Lines in the backtest are SYNTHETIC (rounded baseline means / standard lines), because historical "
                   "PrizePicks/Tailgate lines are not available. Real platform lines contain more information, so "
                   "edges against them will be smaller. Markets that do not beat the baseline are returned as NO BET.")
    market = st.selectbox("Market detail", [r["market"] for r in rows] if rows else [])
    if market:
        bt = load_bt(run_id, market)
        mdl = bt[bt.model_name != "baseline"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                                     name="perfect", hoverinfo="skip"))
            for i, (name, g) in enumerate((("model", mdl), ("baseline", bt[bt.model_name == "baseline"]))):
                cal = backtest_calibration(g, 10)
                if len(cal):
                    fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["observed"], mode="lines+markers", name=name,
                                             line=dict(color=SERIES[i], width=2), marker=dict(size=8),
                                             hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<extra>" + name + "</extra>"))
            fig.update_layout(title=f"P(over line) reliability · {market}", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                              xaxis_title="Predicted", yaxis_title="Observed")
            st.plotly_chart(style_fig(fig, 360), width="stretch")
        with c2:
            by = st.radio("Breakdown", ["surface", "level", "best_of"], horizontal=True)
            t = backtest_summary(mdl, [by])
            cols = [by, "n", "brier", "ece", "mae", "rmse", "bias"]
            st.dataframe(t[[c for c in cols if c in t]].style.format(
                {"brier": "{:.4f}", "ece": "{:.3f}", "mae": "{:.3f}", "rmse": "{:.3f}", "bias": "{:+.3f}"}),
                hide_index=True, width="stretch")
    roi = pd.concat([simulated_flat_roi(load_bt(run_id, r["market"])) for r in rows]) if rows else pd.DataFrame()
    with st.expander("Illustrative flat-stake ROI at -110 on synthetic lines (NOT evidence of real profitability)"):
        st.dataframe(roi, hide_index=True, width="stretch")

with tab_live:
    st.markdown("Performance of recommendations actually saved and graded (verified results only).")
    for by in ("market", "platform", "tour", "month"):
        t = tracked_breakdown(db, by)
        st.markdown(f"**By {by}**")
        if t.empty:
            st.caption("No graded recommendations yet.")
        else:
            st.dataframe(t.style.format({"win_pct": "{:.1%}", "brier": "{:.4f}", "profit": "{:,.2f}", "roi": "{:.1%}"},
                                        na_rep="—"), hide_index=True, width="stretch")

with tab_ver:
    rows = []
    for r in versions.itertuples():
        m = json.loads(r.metrics_json)["markets"].get("match_winner", {})
        rows.append({"version": r.version, "tour": r.tour, "created": r.created_at, "winner model": r.model_type,
                     "trained through": r.train_end, "active": bool(r.is_active), "OOS log loss": m.get("log_loss"),
                     "OOS Brier": m.get("brier"), "OOS accuracy": m.get("accuracy"), "ECE": m.get("ece")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Parameters of selected version"):
        st.json(params)
footer()
