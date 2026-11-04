"""Page 2: Match analysis - player comparison and model predictions."""
from datetime import datetime, time, timedelta, timezone

import ui  # noqa: F401
from ui import SERIES, footer, get_database, now_utc, num, pct, setup_page, status_banner, style_fig

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tennis_edge.db.database import utcnow_iso
from tennis_edge.models.predict import get_predictor

setup_page("Match Analysis", "Compare two players and see the model's predicted outcomes")
db = get_database()
status_banner(db)


@st.cache_data(ttl=3600, show_spinner=False)
def active_players(tour: str) -> pd.DataFrame:
    return get_database().df(
        "SELECT p.player_id, p.full_name, MAX(m.est_match_date) last FROM players p JOIN matches m "
        "ON m.winner_id=p.player_id OR m.loser_id=p.player_id WHERE p.tour=? AND m.est_match_date >= '2024-01-01' "
        "AND m.level NOT IN ('10','15','25','S') GROUP BY p.player_id ORDER BY COUNT(*) DESC", (tour,))


c = st.columns([1, 2, 2, 1, 1, 1.4])
tour = c[0].selectbox("Tour", ["ATP", "WTA"])
players = active_players(tour)
if players.empty:
    st.error("No player data. Run ingestion first.")
    st.stop()
labels = players["full_name"] + "  ·  last " + players["last"]
ia = c[1].selectbox("Player A", range(len(players)), format_func=lambda i: labels.iloc[i], index=0)
ib = c[2].selectbox("Player B", range(len(players)), format_func=lambda i: labels.iloc[i], index=1)
surface = c[3].selectbox("Surface", ["Hard", "Clay", "Grass"])
best_of = c[4].selectbox("Best of", [3, 5])
day = c[5].date_input("Match date", value=now_utc().date() + timedelta(days=1))
a_id, b_id = players["player_id"].iloc[ia], players["player_id"].iloc[ib]
a_name, b_name = players["full_name"].iloc[ia], players["full_name"].iloc[ib]
if a_id == b_id:
    st.warning("Choose two different players.")
    st.stop()
when = datetime.combine(day, time(12, 0), tzinfo=timezone.utc)

pred = get_predictor(db, tour)
if not pred.ready:
    st.error(pred.why_not_ready())
    st.stop()
mp = pred.predict(a_id, b_id, when, surface, int(best_of))
f, q = mp.features, mp.quality

# ------------------------------------------------------------------ data quality
warn = []
for side, nm in (("a", a_name), ("b", b_name)):
    if q.get(f"{side}_days_since_last") and q[f"{side}_days_since_last"] > 60:
        warn.append(f"{nm}: last recorded match {q[f'{side}_last_match']} ({q[f'{side}_days_since_last']:.0f} days before)")
    if q.get(f"{side}_stat_matches_365", 0) < 8:
        warn.append(f"{nm}: only {q[f'{side}_stat_matches_365']:.0f} matches with serve stats in the prior year")
    if not q.get(f"{side}_ranked"):
        warn.append(f"{nm}: not in the latest ranking file ({q['rankings_as_of']})")
if warn:
    st.warning("Data limitations:\n\n" + "\n".join(f"- {w}" for w in warn))

# ------------------------------------------------------------------ comparison
st.subheader("Player comparison")


def record(pid: str, surf: str | None = None, since: str = "2015-01-01") -> str:
    cond = "AND surface=?" if surf else ""
    args = (pid, since) + ((surf,) if surf else ())
    w = db.scalar(f"SELECT COUNT(*) FROM matches WHERE winner_id=? AND est_match_date>=? AND status!='walkover' {cond}", args)
    args2 = (pid, since) + ((surf,) if surf else ())
    lo = db.scalar(f"SELECT COUNT(*) FROM matches WHERE loser_id=? AND est_match_date>=? AND status!='walkover' {cond}", args2)
    return f"{w}-{lo} ({w / (w + lo):.0%})" if (w + lo) else "—"


rows = [
    ("Ranking (latest file)", num(f.get("a_rank"), 0), num(f.get("b_rank"), 0)),
    ("Elo (overall)", num(f["a_elo"], 0), num(f["b_elo"], 0)),
    (f"Elo ({surface})", num(f["a_elo_surf"], 0), num(f["b_elo_surf"], 0)),
    ("Elo (surface-blended)", num(f["a_elo_blend"], 0), num(f["b_elo_blend"], 0)),
    (f"{surface} record since 2015", record(a_id, surface), record(b_id, surface)),
    ("Overall record since 2015", record(a_id), record(b_id)),
    ("Form (Elo residual, decayed)", num(f["a_form"], 3), num(f["b_form"], 3)),
    ("Serve points won (opp-adjusted)", pct(f["a_spw_adj"]), pct(f["b_spw_adj"])),
    ("Return points won (opp-adjusted)", pct(f["a_rpw_adj"]), pct(f["b_rpw_adj"])),
    ("1st serve in", pct(f["a_first_in"]), pct(f["b_first_in"])),
    ("1st serve points won", pct(f["a_first_won"]), pct(f["b_first_won"])),
    ("2nd serve points won", pct(f["a_second_won"]), pct(f["b_second_won"])),
    ("Ace rate (per serve point, surface)", pct(f["a_ace_rate_surf"], 2), pct(f["b_ace_rate_surf"], 2)),
    ("Double-fault rate", pct(f["a_df_rate"], 2), pct(f["b_df_rate"], 2)),
    ("Break-point conversion", pct(f["a_bp_conv"]), pct(f["b_bp_conv"])),
    ("Break points saved", pct(f["a_bp_save"]), pct(f["b_bp_save"])),
    ("Days since last match", num(q.get("a_days_since_last"), 0), num(q.get("b_days_since_last"), 0)),
    ("Matches in prior 14 days", num(f["a_m14"], 0), num(f["b_m14"], 0)),
    ("Matches with stats (prior 365d)", num(f["a_stat_matches_365"], 0), num(f["b_stat_matches_365"], 0)),
]
st.dataframe(pd.DataFrame(rows, columns=["", a_name, b_name]), hide_index=True, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Recent form (last 10 matches)**")
    for pid, nm in ((a_id, a_name), (b_id, b_name)):
        rec = db.df("SELECT est_match_date d, CASE WHEN winner_id=? THEN 'W' ELSE 'L' END r, score, surface, round "
                    "FROM matches WHERE (winner_id=? OR loser_id=?) AND status!='walkover' ORDER BY est_match_date DESC, "
                    "round_order DESC LIMIT 10", (pid, pid, pid))
        st.markdown(f"{nm}: `{' '.join(rec['r'])}`" if len(rec) else f"{nm}: no matches")
with c2:
    st.markdown("**Head-to-head**")
    h2h = db.df("SELECT m.est_match_date date, t.name tournament, m.surface, m.round, p.full_name winner, m.score "
                "FROM matches m JOIN tournaments t ON t.tourney_key=m.tourney_key JOIN players p ON p.player_id=m.winner_id "
                "WHERE (m.winner_id=? AND m.loser_id=?) OR (m.winner_id=? AND m.loser_id=?) ORDER BY m.est_match_date DESC",
                (a_id, b_id, b_id, a_id))
    if h2h.empty:
        st.caption("No previous meetings in the database.")
    else:
        st.dataframe(h2h, hide_index=True, width="stretch", height=200)

# ------------------------------------------------------------------ predictions
st.subheader("Model predictions")
st.caption(f"Model {mp.model_version} · features as of {q['data_through']} evaluated at {day} · "
           f"{q['days_beyond_data']:.0f} days beyond the last historical match")
m = mp.markov
c = st.columns(6)
c[0].metric(f"P({a_name.split()[-1]} wins)", pct(mp.p_a_wins))
c[1].metric("Expected total games", num(m["total_games_mean"][0], 1))
c[2].metric("P(deciding set)", pct(m["sets_pmf"][0][5 if best_of == 5 else 3]))
c[3].metric("1st-set games", num(m["first_set_mean"][0], 1))
c[4].metric(f"Games {a_name.split()[-1]} / {b_name.split()[-1]}",
            f"{m['a_games_mean'][0]:.1f} / {m['b_games_mean'][0]:.1f}")
c[5].metric("Fair odds A / B", f"{1 / mp.p_a_wins:.2f} / {1 / (1 - mp.p_a_wins):.2f}")

c1, c2 = st.columns(2)
with c1:
    pmf = m["total_games_pmf"][0]
    k = np.arange(len(pmf))
    keep = (k >= 12) & (k <= (66 if best_of == 5 else 40))
    fig = go.Figure(go.Bar(x=k[keep], y=pmf[keep], marker_color=SERIES[0], name="P(total games)",
                           hovertemplate="%{x} games: %{y:.1%}<extra></extra>"))
    fig.update_layout(title="Total games distribution", xaxis_title="Total games", yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, legend=False), width="stretch")
with c2:
    rows = []
    for market, label in (("aces", "Aces"), ("double_faults", "Double faults"), ("break_points_won", "Break points won")):
        rows.append({"Stat": label, a_name: round(float(mp.props[market]["a"][0]), 2),
                     b_name: round(float(mp.props[market]["b"][0]), 2)})
    fig = go.Figure()
    for i, nm in enumerate((a_name, b_name)):
        fig.add_trace(go.Bar(x=[r["Stat"] for r in rows], y=[r[nm] for r in rows], name=nm, marker_color=SERIES[i],
                             hovertemplate="%{x}: %{y:.2f}<extra>" + nm + "</extra>"))
    fig.update_layout(title="Expected player statistics", barmode="group", bargap=0.35, bargroupgap=0.08)
    st.plotly_chart(style_fig(fig), width="stretch")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with st.expander("Add this match to upcoming matches (manual schedule entry)"):
    with st.form("sched"):
        tourn = st.text_input("Tournament")
        tstart = st.time_input("Start time (UTC)", value=time(12, 0))
        if st.form_submit_button("Add"):
            start = datetime.combine(day, tstart, tzinfo=timezone.utc)
            if start <= now_utc():
                st.error("Start time must be in the future.")
            else:
                db.execute("INSERT OR IGNORE INTO scheduled_matches(tour,player_a_name,player_b_name,player_a_id,player_b_id,"
                           "start_time,tournament,surface,best_of,source,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (tour, a_name, b_name, a_id, b_id, start.isoformat(), tourn or None, surface, int(best_of),
                            "manual entry", utcnow_iso()))
                db.audit("add_scheduled_match", "scheduled_matches", None, {"a": a_id, "b": b_id}, actor="user")
                st.success("Added.")
footer()
