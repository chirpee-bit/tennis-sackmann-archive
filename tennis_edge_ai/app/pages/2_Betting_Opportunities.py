"""Page 3: Betting opportunities - ranked, with reasons for and against."""
import ui  # noqa: F401
from ui import cached_opportunities, footer, get_database, num, pct, setup_page, status_banner

import pandas as pd
import streamlit as st

from tennis_edge.betting.evaluator import Opportunity
from tennis_edge.config import get_settings
from tennis_edge.markets import load_markets
from tennis_edge.tracking.tracker import TrackingError, save_recommendation

setup_page("Betting Opportunities",
           "Ranked by calibrated edge × data reliability × validated market skill — never by raw confidence")
db = get_database()
status_banner(db)
markets = load_markets()

c = st.columns([2, 3, 1.2, 1.2])
plat = c[0].multiselect("Platform", ["PrizePicks", "Tailgate", "Sportsbook"], default=[])
mk = c[1].multiselect("Market", list(markets), format_func=lambda k: markets[k].label)
show_nobet = c[2].toggle("Show NO BET", value=True)
min_conf = c[3].selectbox("Min. confidence", ["Any", "B", "A"])
if st.button("↻ Re-evaluate now"):
    st.cache_data.clear()

opps = cached_opportunities(str(db.path), tuple(plat), tuple(mk))
if not opps:
    st.info("No current projections match. Import PrizePicks or Tailgate projections on the Data Management page.")
    footer()
    st.stop()
rows = [o for o in opps if show_nobet or o["is_bet"]]
if min_conf != "Any":
    rows = [o for o in rows if o["confidence"] in (("A",) if min_conf == "A" else ("A", "B"))]
n_bet = sum(o["is_bet"] for o in opps)
st.markdown(f"**{len(opps)}** projections evaluated · **{n_bet}** qualify as bets · threshold: edge lower bound ≥ "
            f"{get_settings().min_edge:.0%}")
if not rows:
    st.warning("Nothing to show with these filters. NO BET is a valid outcome.")
    footer()
    st.stop()

df = pd.DataFrame(rows)
table = pd.DataFrame({
    "#": range(1, len(df) + 1), "Player": df["player_name"], "Opponent": df["opponent_name"],
    "Platform": df["platform"], "Market": df["market_label"], "Line": df["line"], "Side": df["side"],
    "Model P": df["model_prob"].map(pct), "Market P": df["implied_prob"].map(pct),
    "Break-even": df["breakeven_prob"].map(pct), "Edge": df["edge"].map(pct), "Edge (low)": df["edge_lower"].map(pct),
    "EV/unit": df["ev_per_unit"].map(lambda x: num(x, 3)), "Fair odds": df["fair_odds_american"].map(lambda x: num(x, 0)),
    "Proj. diff": df["projection_diff"].map(lambda x: num(x, 2)), "±": df["uncertainty"].map(pct),
    "Data age (d)": df["data_age_days"].map(lambda x: num(x, 0)), "Sample": df["sample_size"].map(lambda x: num(x, 0)),
    "Conf.": df["confidence"], "Score": df["rank_score"].map(lambda x: num(x, 2)),
})
st.dataframe(table, hide_index=True, width="stretch", height=min(560, 38 + 35 * len(table)))
st.caption("Market P = margin-free probability from two-way odds (power method) when the platform lists prices. "
           "Break-even = per-leg probability needed by a VERIFIED pick'em payout table. EV for pick'em assumes all legs "
           "of the entry share this probability and are independent. ± = heuristic uncertainty.")

st.subheader("Details")
for i, o in enumerate(rows[:40]):
    title = (f"{i + 1}. {o['player_name']} vs {o['opponent_name'] or '?'} — {o['market_label']} "
             f"{o['side'] or ''} {o['line'] if o['line'] is not None else ''} · {o['platform']} · {o['confidence']}")
    with st.expander(title):
        a, b = st.columns(2)
        with a:
            st.markdown("**Reasons for**")
            st.markdown("\n".join(f"- {r}" for r in o["reasons_for"]) or "—")
        with b:
            st.markdown("**Reasons against**")
            st.markdown("\n".join(f"- {r}" for r in o["reasons_against"]) or "—")
        if o["no_bet_reasons"]:
            st.error("NO BET: " + " · ".join(o["no_bet_reasons"]))
        st.caption(f"P(over) {pct(o['prob_over'])} · P(under) {pct(o['prob_under'])} · P(push) {pct(o['prob_push'])} · "
                   f"model mean {o['model_mean']} · {o['implied_note']} · {o['ev_note']} · model {o['model_version']}")
        if o["is_bet"]:
            with st.form(f"save_{i}"):
                stake = st.number_input("Stake (0 = paper tracking only)", min_value=0.0, value=0.0, step=1.0)
                if st.form_submit_button("Save to tracker"):
                    try:
                        rid = save_recommendation(db, Opportunity(**o), stake=stake, actor="user")
                        st.success(f"Saved as recommendation #{rid} at the current line.")
                    except TrackingError as exc:
                        st.error(str(exc))
footer()
