"""Phase 3: build leakage-free pre-match features and the engine snapshot per tour."""
import argparse
import time

import _bootstrap  # noqa: F401

from tennis_edge.db.database import get_db
from tennis_edge.features.builder import build_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["ATP", "WTA"])
    ap.add_argument("--emit-from", type=int, default=2005)
    a = ap.parse_args()
    db = get_db()
    for tour in a.tours:
        t0 = time.time()
        feats, eng = build_features(db, tour, a.emit_from)
        print(f"{tour}: {len(feats):,} feature rows, {feats.shape[1]} columns, engine players kept="
              f"{len(eng.players):,}, last match date index={eng.last_t:.0f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
