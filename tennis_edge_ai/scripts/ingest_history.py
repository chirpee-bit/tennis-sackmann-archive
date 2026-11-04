"""Phase 2: load the Sackmann historical archive into SQLite.

Usage:  python scripts/ingest_history.py [--tours ATP WTA] [--no-rankings]
Idempotent: re-running only inserts new rows.
"""
import argparse
import time

import _bootstrap  # noqa: F401

from tennis_edge.db.database import get_db
from tennis_edge.ingest.historical import ingest_rankings, ingest_tour
from tennis_edge.providers.sackmann import SackmannProvider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["ATP", "WTA"])
    ap.add_argument("--no-rankings", action="store_true")
    a = ap.parse_args()
    db = get_db()
    provider = SackmannProvider()
    st = provider.status()
    print(f"Provider: {st.name} -> {'AVAILABLE' if st.available else 'UNAVAILABLE'}: {st.reason}")
    if not st.available:
        raise SystemExit(1)
    for tour in a.tours:
        t0 = time.time()
        rep = ingest_tour(db, tour, provider)
        print(f"{tour}: read={rep.rows_read:,} dup_dropped={rep.duplicates_dropped:,} invalid={rep.invalid_dropped:,} "
              f"inserted={rep.matches_inserted:,} stat_rows={rep.stats_inserted:,} players={rep.players_upserted:,} "
              f"status={rep.status_counts} ({time.time()-t0:.0f}s)")
        if not a.no_rankings:
            print(f"{tour}: rankings rows inserted={ingest_rankings(db, tour, provider):,}")
    print(db.table_counts())


if __name__ == "__main__":
    main()
