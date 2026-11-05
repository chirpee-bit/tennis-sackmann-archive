"""Page 4: Tracker - recommendations, grading, pick'em entries, bankroll, closing lines."""
import json

import ui  # noqa: F401
from ui import footer, get_database, money, pct, setup_page

import pandas as pd
import streamlit as st

from tennis_edge.analytics.performance import tracked_summary
from tennis_edge.betting.pickem import load_payouts
from tennis_edge.tracking.tracker import (TrackingError, add_transaction, auto_grade, capture_closing_lines,
                                          create_entry, grade)

setup_page("Tracker", "Every recommendation is stored before the match with its exact line, price and timestamp")
db = get_database()

s = tracked_summary(db)
c = st.columns(6)
c[0].metric("Recommendations", s["total_recommendations"])
c[1].metric("Record (W-L-P-V)", f"{s['win']}-{s['loss']}-{s['push']}-{s['void']}")
c[2].metric("Win %", pct(s["win_pct"]))
c[3].metric("Profit / loss", money(s["profit"]))
c[4].metric("ROI", pct(s["roi"]))
c[5].metric("Avg closing-line value", "—" if s["avg_clv_line"] is None else f"{s['avg_clv_line']:+.2f} ({s['clv_n']})",
            help="Line points by which the closing line moved in our favour (over: closing line higher)")

tab_recs, tab_grade, tab_entries, tab_bank = st.tabs(["Recommendations", "Grade results", "Pick'em entries", "Bankroll"])
with tab_recs:
    status = st.multiselect("Status", ["pending", "win", "loss", "push", "void"], default=[])
    df = db.df("SELECT r.id, r.created_at, r.platform, r.market, r.player_name, r.opponent_name, r.line, r.side, "
               "r.odds_american, r.payout_structure, r.model_prob, r.edge, r.confidence, r.stake, r.start_time, r.status, "
               "r.result_value, s.profit, r.grade_source, r.closing_line, r.closing_odds, r.model_version "
               "FROM bet_recommendations r LEFT JOIN settled_results s ON s.recommendation_id=r.id ORDER BY r.created_at DESC")
    if status:
        df = df[df["status"].isin(status)]
    if df.empty:
        st.info("No recommendations saved yet. Save them from the Betting Opportunities page (or let the scheduler do it).")
    else:
        st.dataframe(df, hide_index=True, width="stretch", height=420)
        st.download_button("Export CSV", df.to_csv(index=False).encode(), "recommendations.csv", "text/csv")

with tab_grade:
    a, b = st.columns(2)
    if a.button("Auto-grade from verified historical results"):
        res = auto_grade(db)
        st.success(f"Graded {res['graded']} · waiting for verified results: {res['waiting']}")
    if b.button("Capture closing lines"):
        st.success(f"Closing lines captured: {capture_closing_lines(db)}")
    st.caption("Auto-grading only uses results present in the historical database (matched by both players and date). "
               "Walkovers and retirements are graded VOID by default (TE_RETIREMENT_RULE) — check your platform's rules.")
    pending = db.query("SELECT id, player_name, opponent_name, market, side, line, start_time FROM bet_recommendations "
                       "WHERE status='pending' ORDER BY start_time")
    if pending:
        with st.form("manual_grade"):
            st.markdown("**Manual grading** (record the verified source)")
            rid = st.selectbox("Recommendation", [p["id"] for p in pending],
                               format_func=lambda i: next(f"#{p['id']} {p['player_name']} vs {p['opponent_name']} · "
                                                          f"{p['market']} {p['side']} {p['line']} · {p['start_time'][:16]}"
                                                          for p in pending if p["id"] == i))
            mode = st.radio("Enter", ["Final stat value", "Outcome"], horizontal=True)
            val = st.number_input("Final value (e.g. total games; 1/0 for match winner)", value=0.0, step=1.0)
            outcome = st.selectbox("Outcome", ["win", "loss", "push", "void"])
            source = st.text_input("Verified source (required)", placeholder="e.g. official ATP match stats page")
            if st.form_submit_button("Grade"):
                if not source.strip():
                    st.error("A verified source is required.")
                else:
                    try:
                        r = grade(db, int(rid), outcome if mode == "Outcome" else None,
                                  None if mode == "Outcome" else float(val), source=f"manual:{source.strip()}")
                        st.success(f"Graded #{rid}: {r['outcome']} (profit {money(r['profit'])})")
                    except TrackingError as exc:
                        st.error(str(exc))
    else:
        st.caption("No pending recommendations.")

with tab_entries:
    structures = load_payouts()
    pend = db.query("SELECT id, platform, player_name, market, side, line, start_time FROM bet_recommendations "
                    "WHERE status='pending' AND odds_american IS NULL")
    with st.form("entry"):
        st.markdown("**Record a pick'em entry** (you place it yourself on the platform; this only tracks it)")
        sid = st.selectbox("Payout structure", list(structures),
                           format_func=lambda k: f"{structures[k].label} — {'VERIFIED' if structures[k].verified else 'UNVERIFIED'}")
        legs = st.multiselect("Legs", [p["id"] for p in pend],
                              format_func=lambda i: next(f"#{p['id']} {p['platform']} {p['player_name']} {p['market']} "
                                                         f"{p['side']} {p['line']}" for p in pend if p["id"] == i))
        stake = st.number_input("Stake", min_value=0.0, value=10.0, step=1.0)
        if st.form_submit_button("Save entry"):
            try:
                eid = create_entry(db, [int(x) for x in legs], sid, stake)
                st.success(f"Entry #{eid} saved with a snapshot of the payout table.")
            except TrackingError as exc:
                st.error(str(exc))
    ent = db.df("SELECT * FROM pickem_entries ORDER BY created_at DESC")
    if len(ent):
        ent["payout_table"] = ent["payout_snapshot"].map(lambda x: json.loads(x)["payouts"])
        st.dataframe(ent.drop(columns=["payout_snapshot"]), hide_index=True, width="stretch")

with tab_bank:
    with st.form("bank"):
        k = st.selectbox("Type", ["deposit", "withdrawal", "adjustment"])
        amt = st.number_input("Amount", value=0.0, step=10.0)
        note = st.text_input("Note")
        if st.form_submit_button("Add transaction"):
            add_transaction(db, k, amt, note)
            st.success("Recorded.")
    st.caption("No sportsbook or platform passwords are stored. Balances here are your own records only.")
    bank = db.df("SELECT * FROM bankroll_transactions ORDER BY ts DESC")
    st.metric("Ledger balance", money(float(bank["amount"].sum()) if len(bank) else 0.0))
    if len(bank):
        st.dataframe(bank, hide_index=True, width="stretch")
footer()
