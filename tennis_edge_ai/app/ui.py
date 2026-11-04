"""Shared Streamlit helpers: path setup, styling, cached resources."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from tennis_edge.config import get_settings  # noqa: E402
from tennis_edge.db.database import Database  # noqa: E402

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
              "This tool never places wagers. Historical data: Jeff Sackmann (CC BY-NC-SA 4.0), non-commercial use.")


def setup_page(title: str) -> None:
    st.set_page_config(page_title=f"TENNIS EDGE AI · {title}", page_icon="🎾", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)


@st.cache_resource
def get_database() -> Database:
    return Database(get_settings().db_path)


def badge(text: str, kind: str = "warn") -> str:
    return f'<span class="te-badge te-{kind}">{text}</span>'


def footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
