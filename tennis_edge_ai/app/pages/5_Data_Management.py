"""Page 6: Data management - uploads, providers, validation, payout tables, backups."""
from datetime import datetime, time, timedelta, timezone

import ui  # noqa: F401
from ui import footer, get_database, now_utc, setup_page

import pandas as pd
import streamlit as st

from tennis_edge.automation.jobs import generate_recommendations, grade_and_close, import_dropbox, refresh_sources
from tennis_edge.betting.pickem import breakeven_leg_prob, load_payouts, save_verified_flag, upsert_structure
from tennis_edge.config import get_settings
from tennis_edge.markets import load_markets, set_platform_verified
from tennis_edge.providers.projections_csv import CSVProjectionProvider, ManualEntryProvider
from tennis_edge.providers.registry import all_providers

setup_page("Data Management", "Imports, provider status, validation, payout tables and backups")
db = get_database()
settings = get_settings()

TEMPLATE = ("platform,player,opponent,market,line,over_odds,under_odds,start_time,tournament,surface,tour,best_of,"
            "payout_structure\n"
            "PrizePicks,Jannik Sinner,Carlos Alcaraz,Total Games,38.5,,,2026-11-08T14:00:00Z,ATP Finals,Hard,ATP,3,PP_POWER_2\n"
            "Tailgate,Iga Swiatek,Coco Gauff,Aces,2.5,-115,-105,2026-11-08T18:00:00Z,WTA Finals,Hard,WTA,3,\n")

t_up, t_prov, t_val, t_pay, t_jobs, t_db = st.tabs(
    ["CSV upload / entry", "Providers & API", "Data validation", "Payouts & markets", "Jobs", "Backup & export"])

with t_up:
    st.markdown("Upload projections you have obtained legitimately (typed in, or exported from a source you are "
                "licensed to use). The app never scrapes PrizePicks, Tailgate or any sportsbook.")
    st.download_button("Download CSV template", TEMPLATE.encode(), "projections_template.csv", "text/csv")
    platform = st.selectbox("Platform for this file", ["PrizePicks", "Tailgate"])
    f = st.file_uploader("Projection CSV", type=["csv"])
    if f is not None and st.button("Validate & import"):
        try:
            res = CSVProjectionProvider(platform).import_bytes(db, f.getvalue(), f.name)
            st.success(f"Valid rows: {res['valid_rows']} · new {res['inserted']} · updated {res['updated']} · "
                       f"unchanged {res['unchanged']} · errors {len(res['errors'])}")
            if res["errors"]:
                st.error("Rejected rows")
                st.dataframe(pd.DataFrame(res["errors"]), hide_index=True, width="stretch")
            if res["unresolved"]:
                st.warning("Names not matched to historical players (these projections will be NO BET):\n\n" +
                           "\n".join(f"- {u}" for u in res["unresolved"]))
            if res["line_moves"]:
                st.info(f"{len(res['line_moves'])} significant line moves detected and logged as alerts.")
            st.cache_data.clear()
        except ValueError as exc:
            st.error(f"File rejected: {exc}")
    st.divider()
    with st.form("manual"):
        st.markdown("**Manual entry**")
        markets = load_markets()
        c = st.columns(4)
        mplat = c[0].selectbox("Platform", ["PrizePicks", "Tailgate"])
        mtour = c[1].selectbox("Tour", ["ATP", "WTA"])
        mmkt = c[2].selectbox("Market", list(markets), format_func=lambda k: markets[k].label)
        mline = c[3].number_input("Line", value=21.5, step=0.5)
        c = st.columns(4)
        mp = c[0].text_input("Player")
        mo = c[1].text_input("Opponent")
        msurf = c[2].selectbox("Surface", ["Hard", "Clay", "Grass"])
        mbo = c[3].selectbox("Best of", [3, 5])
        c = st.columns(4)
        mdate = c[0].date_input("Date (UTC)", value=now_utc().date() + timedelta(days=1))
        mtime = c[1].time_input("Time (UTC)", value=time(14, 0))
        oo = c[2].text_input("Over odds (American, optional)")
        uo = c[3].text_input("Under odds (American, optional)")
        if st.form_submit_button("Add projection"):
            row = {"player": mp, "opponent": mo, "market": markets[mmkt].label, "line": mline,
                   "over_odds": oo, "under_odds": uo, "tour": mtour, "surface": msurf, "best_of": mbo,
                   "start_time": datetime.combine(mdate, mtime, tzinfo=timezone.utc).isoformat()}
            res = ManualEntryProvider(mplat).import_rows(db, [row])
            if res["errors"]:
                st.error("; ".join(e["error"] for e in res["errors"]))
            else:
                st.success("Projection stored." + (f" Unmatched: {res['unresolved']}" if res["unresolved"] else ""))
                st.cache_data.clear()
    cur = db.df("SELECT platform, market, tour, player_name, opponent_name, line, over_odds, under_odds, start_time, "
                "first_line, fetched_at, player_id IS NOT NULL AS player_matched, opponent_id IS NOT NULL AS opp_matched "
                "FROM projections_current ORDER BY start_time")
    st.markdown(f"**Current projections ({len(cur)})**")
    st.dataframe(cur, hide_index=True, width="stretch")

with t_prov:
    st.dataframe(pd.DataFrame([p.status().as_dict() for p in all_providers()]), hide_index=True, width="stretch")
    st.markdown("**Configuration** (secrets are masked; edit the `.env` file on the server — keys are never entered "
                "or displayed here)")
    st.dataframe(pd.DataFrame(settings.public_view().items(), columns=["setting", "value"]), hide_index=True,
                 width="stretch")
    st.markdown("""
| Source | What you need |
|---|---|
| Historical results | Nothing (bundled Sackmann archive). To update: pull `JeffSackmann/tennis_atp` and `tennis_wta`, then re-run ingestion, features and training. |
| PrizePicks | No public API exists. Use CSV / manual entry, or a written data licence from PrizePicks or an authorised reseller (`PRIZEPICKS_LICENSED_API_URL`/`_KEY`). |
| Tailgate | CSV / manual entry, or authorised API credentials (`TAILGATE_API_URL`/`_KEY`). |
| Sportsbook odds (optional) | `ODDS_API_KEY` from the-odds-api.com (free tier ≈ 500 requests/month; paid plans for more). |
""")

with t_val:
    st.markdown("**Coverage by season (matches with serve statistics)**")
    cov = db.df("SELECT m.tour, substr(m.est_match_date,1,4) season, COUNT(*) matches, "
                "SUM(CASE WHEN s.match_id IS NOT NULL THEN 1 ELSE 0 END) with_stats, "
                "SUM(CASE WHEN m.status!='completed' THEN 1 ELSE 0 END) not_completed "
                "FROM matches m LEFT JOIN (SELECT DISTINCT match_id FROM match_stats) s ON s.match_id=m.match_id "
                "WHERE m.est_match_date >= '2015-01-01' GROUP BY m.tour, season ORDER BY m.tour, season")
    cov["stats_coverage"] = (cov["with_stats"] / cov["matches"]).map(lambda x: f"{x:.0%}")
    st.dataframe(cov, hide_index=True, width="stretch")
    last = db.scalar("SELECT MAX(est_match_date) FROM matches")
    if last:
        age = (now_utc().date() - datetime.fromisoformat(last).date()).days
        (st.error if age > settings.max_data_age_days else st.warning if age > 14 else st.success)(
            f"Latest historical match: {last} ({age} days ago).")
    warns = []
    n_unk = db.scalar("SELECT COUNT(*) FROM matches WHERE status='unknown' AND est_match_date >= '2020-01-01'")
    warns.append(f"{n_unk:,} matches since 2020 with unparseable/non-standard scores (excluded from game targets)")
    n_np = db.scalar("SELECT COUNT(*) FROM projections_current WHERE player_id IS NULL OR "
                     "(opponent_name IS NOT NULL AND opponent_id IS NULL)")
    warns.append(f"{n_np} current projections with unmatched player names")
    n_ns = db.scalar("SELECT COUNT(*) FROM projections_current WHERE surface IS NULL")
    warns.append(f"{n_ns} current projections without surface (Hard assumed, flagged)")
    st.markdown("**Missing-data warnings**\n" + "\n".join(f"- {w}" for w in warns))
    st.markdown("**Data source log**")
    st.dataframe(db.df("SELECT * FROM data_source_logs ORDER BY id DESC LIMIT 100"), hide_index=True, width="stretch")
    st.markdown("**Alerts**")
    st.dataframe(db.df("SELECT ts, kind, message, delivered FROM alerts ORDER BY id DESC LIMIT 100"), hide_index=True,
                 width="stretch")
    st.markdown("**Audit log**")
    st.dataframe(db.df("SELECT * FROM audit_log ORDER BY id DESC LIMIT 200"), hide_index=True, width="stretch")

with t_pay:
    st.markdown("Expected value for pick'em is computed **only** from payout tables you mark as verified against your "
                "own app (payouts vary by platform, state and date).")
    structs = load_payouts()
    rows = [{"id": s.id, "platform": s.platform, "label": s.label, "legs": s.legs,
             "payouts (hits→multiplier)": ", ".join(f"{k}→{v}x" for k, v in sorted(s.payouts.items(), reverse=True)),
             "break-even per leg": f"{breakeven_leg_prob(s):.1%}", "verified": s.verified} for s in structs.values()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.form("verify"):
        sid = st.selectbox("Structure", list(structs))
        ver = st.checkbox("I have confirmed these exact multipliers in my app today")
        if st.form_submit_button("Save verification status"):
            save_verified_flag(sid, ver)
            db.audit("payout_verified", "payouts", sid, {"verified": ver}, actor="user")
            st.success("Saved.")
            st.cache_data.clear()
    with st.form("new_struct"):
        st.markdown("**Add / replace a payout structure**")
        c = st.columns(4)
        nid = c[0].text_input("Id (e.g. PP_POWER_4)")
        nplat = c[1].selectbox("Platform ", ["PrizePicks", "Tailgate"])
        nlegs = c[2].number_input("Legs", min_value=2, max_value=8, value=2)
        nvoid = c[3].selectbox("Void rule", ["drop_leg", "refund"])
        npay = st.text_input("Payouts as hits:multiplier, comma separated (e.g. 3:2.25, 2:1.25)")
        nver = st.checkbox("Verified")
        if st.form_submit_button("Save structure"):
            try:
                pay = {int(a.split(":")[0]): float(a.split(":")[1]) for a in npay.split(",") if a.strip()}
                upsert_structure(nid.strip(), nplat, nid.strip(), int(nlegs), pay, nvoid, nver)
                db.audit("payout_upsert", "payouts", nid, {"payouts": pay, "verified": nver}, actor="user")
                st.success("Saved.")
            except (ValueError, IndexError) as exc:
                st.error(f"Invalid structure: {exc}")
    st.markdown("**Markets verified per platform**")
    mk = load_markets()
    st.dataframe(pd.DataFrame([{"market": m.label, **m.platforms} for m in mk.values()]), hide_index=True, width="stretch")
    with st.form("mkver"):
        c = st.columns(3)
        mkey = c[0].selectbox("Market", list(mk), format_func=lambda k: mk[k].label)
        mpl = c[1].selectbox("Platform", ["PrizePicks", "Tailgate", "Sportsbook"])
        mv = c[2].checkbox("Offered (verified by me)")
        if st.form_submit_button("Save"):
            set_platform_verified(mkey, mpl, mv)
            db.audit("market_verified", "markets", mkey, {"platform": mpl, "verified": mv}, actor="user")
            st.success("Saved.")

with t_jobs:
    st.markdown("Run automation jobs manually (the scheduler runs them on `TE_REFRESH_MINUTES` / `TE_GRADE_MINUTES`). "
                "No job ever places a wager.")
    c = st.columns(4)
    if c[0].button("Refresh configured sources"):
        st.json(refresh_sources(db))
    if c[1].button("Import CSVs from var/imports/"):
        st.json(import_dropbox(db))
    if c[2].button("Generate & store recommendations"):
        st.json(generate_recommendations(db))
        st.cache_data.clear()
    if c[3].button("Closing lines + auto-grade"):
        st.json(grade_and_close(db))

with t_db:
    if st.button("Create database backup"):
        path = db.backup()
        st.success(f"Backup written to {path}")
    tables = list(db.table_counts().items())
    st.dataframe(pd.DataFrame(tables, columns=["table", "rows"]), hide_index=True, width="stretch")
    exportable = [t for t, n in tables if n <= 300000]
    tsel = st.selectbox("Export table to CSV", exportable)
    if tsel:
        st.download_button(f"Download {tsel}.csv", db.df(f'SELECT * FROM "{tsel}"').to_csv(index=False).encode(),
                           f"{tsel}.csv", "text/csv")
footer()
