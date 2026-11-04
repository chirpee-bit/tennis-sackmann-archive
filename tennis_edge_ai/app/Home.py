"""TENNIS EDGE AI - Overview page (Streamlit entry point)."""
import ui  # noqa: F401  (sets sys.path)
from ui import footer, get_database, setup_page

import streamlit as st

setup_page("Overview")
db = get_database()
counts = db.table_counts()
c1, c2, c3 = st.columns(3)
c1.metric("Historical matches", f"{counts.get('matches', 0):,}")
c2.metric("Recommendations", f"{counts.get('bet_recommendations', 0):,}")
c3.metric("Current projections", f"{counts.get('projections_current', 0):,}")
if counts.get("matches", 0) == 0:
    st.warning("No historical data ingested yet. Run `python scripts/ingest_history.py`.")
footer()
