"""Shared Streamlit helpers: path setup, styling, chart theme, cached resources."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from tennis_edge.config import get_settings  # noqa: E402
from tennis_edge.db.database import Database  # noqa: E402

# Categorical series colours, assigned in FIXED order (validated for the dark surface, CVD-safe).
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
SURFACE = "#0E1117"
GRID = "#262C3A"
TEXT = "#E6E9EF"
MUTED = "#9AA4B2"
STATUS = {"good": "#3DDC97", "warning": "#F5B94A", "critical": "#FF6B6B"}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
div[data-testid="stMetric"] {background:#161B26;border:1px solid #232A3A;border-radius:10px;padding:10px 14px;}
div[data-testid="stMetricLabel"] p {color:#9AA4B2;font-size:0.8rem;text-transform:uppercase;letter-spacing:.04em;}
.te-badge {display:inline-block;padding:2px 8px;border-radius:6px;font-size:0.75rem;font-weight:600;margin-right:6px;}
.te-warn {background:#3A2A10;color:#F5B94A;border:1px solid #5A4318;}
.te-ok {background:#0F2E22;color:#3DDC97;border:1px solid #1C4D38;}
.te-bad {background:#3A1416;color:#FF6B6B;border:1px solid #5A2023;}
.te-muted {color:#8A93A3;font-size:0.85rem;}
</style>
"""

DISCLAIMER = ("Statistical estimates only — not financial advice and no guarantee of winnings. "
              "This tool never places wagers or accesses betting accounts. Historical data: Jeff Sackmann "
              "(CC BY-NC-SA 4.0), non-commercial use only.")


def setup_page(title: str, subtitle: str | None = None) -> None:
    st.set_page_config(page_title=f"TENNIS EDGE AI · {title}", page_icon="🎾", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("### 🎾 TENNIS EDGE AI")
    st.sidebar.caption("Statistical tennis projections · tracking · validation")
    st.title(title)
    if subtitle:
        st.caption(subtitle)


@st.cache_resource
def get_database() -> Database:
    return Database(get_settings().db_path)


def badge(text: str, kind: str = "warn") -> str:
    return f'<span class="te-badge te-{kind}">{text}</span>'


def footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)


def pct(x, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.{nd}f}%"


def num(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.{nd}f}"


def money(x) -> str:
    return "—" if x is None else f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def style_fig(fig, height: int = 340, legend: bool = True):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(color=TEXT, size=12), margin=dict(l=48, r=16, t=40, b=40), showlegend=legend,
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#161B26", font_color=TEXT))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


@st.cache_data(ttl=60, show_spinner=False)
def cached_opportunities(_db_path: str, platforms: tuple, markets: tuple, nonce: int = 0) -> list[dict]:
    from tennis_edge.betting.evaluator import evaluate_current
    db = get_database()
    opps = evaluate_current(db, list(platforms) or None, list(markets) or None)
    return [o.to_dict() for o in opps]


def data_status(db: Database) -> dict:
    last = db.scalar("SELECT MAX(est_match_date) FROM matches")
    models = db.query("SELECT tour, version, created_at, train_end FROM model_versions WHERE is_active=1")
    days = None
    if last:
        days = (now_utc().date() - datetime.fromisoformat(last).date()).days
    return {"last_match": last, "days_stale": days, "models": models}


def status_banner(db: Database) -> None:
    s = data_status(db)
    if not s["last_match"]:
        st.error("No historical data ingested. Run `python scripts/ingest_history.py`.")
        return
    limit = get_settings().max_data_age_days
    kind = "ok" if s["days_stale"] <= 14 else ("warn" if s["days_stale"] <= limit else "bad")
    msg = (f"Historical results through <b>{s['last_match']}</b> ({s['days_stale']} days ago). ")
    if kind != "ok":
        msg += ("Recent form, injuries and results after that date are <b>not</b> in the model; "
                "opportunities for players without recent data are returned as NO BET. "
                "Update the data (README → Updating data).")
    models = ", ".join(f"{m['tour']}: {m['version']}" for m in s["models"]) or "none trained"
    st.markdown(f"{badge('DATA', kind)} {msg}<br>{badge('MODELS', 'ok' if s['models'] else 'bad')} {models}",
                unsafe_allow_html=True)
