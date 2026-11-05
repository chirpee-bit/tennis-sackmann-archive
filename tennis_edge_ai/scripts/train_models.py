"""Phases 3-4: walk-forward validation of every market, then fit + register production models.

Usage: python scripts/train_models.py --tours ATP WTA [--no-lgbm] [--first-test-year 2019]
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401

from tennis_edge.db.database import get_db
from tennis_edge.models.training import run_tour


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tours", nargs="+", default=["ATP", "WTA"])
    ap.add_argument("--no-lgbm", action="store_true")
    ap.add_argument("--first-test-year", type=int, default=None)
    a = ap.parse_args()
    db = get_db()
    for tour in a.tours:
        t0 = time.time()
        res = run_tour(db, tour, first_test_year=a.first_test_year, include_lgbm=not a.no_lgbm,
                       progress=lambda m: print(f"{time.time() - t0:7.0f}s {m}", flush=True))
        print(f"\n=== {tour} -> active version {res['version']} (winner model: {res['choice']})")
        print(json.dumps(res["metrics"], indent=1, default=float))
        print("gbm vs logistic:", json.dumps(res["evidence"]["gbm_vs_logistic"], default=float))


if __name__ == "__main__":
    main()
