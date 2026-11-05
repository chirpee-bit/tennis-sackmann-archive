"""TENNIS EDGE AI - Page 1: Overview (Streamlit entry point)."""
import json

import ui  # noqa: F401  (sets sys.path)
from ui import (cached_opportunities, footer, get_database, money, now_utc, num, pct, setup_page, status_banner)

import pandas as pd
import streamlit as st

from tennis_edge.analytics.performance import tracked_summary
from tennis_edge.config import get_settings

setup_page("Overview", "Predictions, verified results and the current best opportunities")
db = get_database()
status_banner(db)

summary = tracked_summary(db)
n_preds = (db.scalar("SELECT COUNT(*) FROM model_predictions") or 0)
active = db.query("SELECT tour, metrics_json FROM model_versions WHERE is_active=1")
mw_ece, mw_brier = [], []
for a in active:
    m = json.loads(a["metrics_json"] or "{}").get("markets", {}).get("match_winner", {})
    if m:
        mw_ece.append(f"{a['tour']} {m.get('ece', 0):.3f}")
        mw_brier.append(m.get("brier"))

c = st.columns(6)
c[0].metric("Total predictions", f"{n_preds:,}", help="Every projection evaluated (incl. NO BET) is logged")
c[1].metric("Recommendations", f"{summary['total_recommendations']:,}", f"{summary['pending']} pending",
            delta_color="off")
c[2].metric("Verified win %", pct(summary["win_pct"]),
            f"{summary['win']}W-{summary['loss']}L-{summary['push']}P-{summary['void']}V", delta_color="off")
c[3].metric("Profit / loss", money(summary["profit"]), f"staked {money(summary['staked'])}", delta_color="off")
c[4].metric("ROI", pct(summary["roi"]))
c[5].metric("Calibration (ECE, match winner)", " · ".join(mw_ece) or "—",
            help="Expected calibration error of out-of-sample walk-forward predictions")
if summary["win"] + summary["loss"] < 30:
    st.caption("Fewer than 30 decided recommendations — win % and ROI are not yet statistically meaningful.")

st.subheader("Upcoming matches")
now = now_utc().isoformat()
up = db.df("SELECT DISTINCT tour, player_name AS player, opponent_name AS opponent, start_time, tournament, surface, "
           "platform FROM projections_current WHERE start_time > ? AND opponent_name IS NOT NULL ORDER BY start_time",
           (now,))
sched = db.df("SELECT tour, player_a_name AS player, player_b_name AS opponent, start_time, tournament, surface, "
              "source AS platform FROM scheduled_matches WHERE start_time > ? ORDER BY start_time", (now,))
upcoming = pd.concat([up, sched], ignore_index=True)
if upcoming.empty:
    st.info("No upcoming matches known. There is no licensed schedule feed configured — matches appear here when you "
            "import projections (Data Management) or add a match on the Match Analysis page.")
else:
    upcoming["pair"] = upcoming.apply(lambda r: " vs ".join(sorted([str(r.player), str(r.opponent)])), axis=1)
    view = upcoming.drop_duplicates(["pair", "start_time"]).drop(columns=["pair", "player"]).rename(
        columns={"opponent": "vs"})
    view.insert(1, "match", upcoming.drop_duplicates(["pair", "start_time"])["pair"].to_numpy())
    st.dataframe(view.head(25), hide_index=True, width="stretch")

st.subheader("Best opportunities")
opps = cached_opportunities(str(db.path), (), ())
bets = [o for o in opps if o["is_bet"]]
if not opps:
    st.info("No projections imported yet. Upload PrizePicks/Tailgate CSVs on the Data Management page.")
elif not bets:
    st.warning(f"NO BET — none of the {len(opps)} current projections passes the filters "
               f"(edge lower bound ≥ {get_settings().min_edge:.0%}, fresh data, sufficient sample, validated market).")
    reasons = pd.Series([r for o in opps for r in o["no_bet_reasons"][:1]]).value_counts().head(5)
    st.markdown("**Most common reasons:**\n" + "\n".join(f"- {k} ({v})" for k, v in reasons.items()))
else:
    df = pd.DataFrame(bets[:10])
    show = pd.DataFrame({
        "Player": df["player_name"], "Opponent": df["opponent_name"], "Platform": df["platform"],
        "Market": df["market_label"], "Line": df["line"], "Side": df["side"],
        "Model P": df["model_prob"].map(pct), "Edge": df["edge"].map(pct),
        "EV/unit": df["ev_per_unit"].map(lambda x: num(x, 3)), "Conf.": df["confidence"],
        "Score": df["rank_score"].map(lambda x: num(x, 2))})
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Details, reasons for/against and saving to the tracker: Betting Opportunities page.")
footer()
