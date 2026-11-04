"""Run the automated jobs on a schedule.

    python -m tennis_edge.automation.scheduler          # runs until stopped
    python -m tennis_edge.automation.scheduler --once   # run every job once and exit

Intervals: TE_REFRESH_MINUTES (refresh + dropbox + recommendations) and
TE_GRADE_MINUTES (closing lines + grading).
"""
from __future__ import annotations

import argparse
import json
import logging

from tennis_edge.automation.jobs import generate_recommendations, grade_and_close, import_dropbox, refresh_sources, run_all
from tennis_edge.config import get_settings
from tennis_edge.db.database import get_db

log = logging.getLogger("tennis_edge.scheduler")


def _cycle():
    db = get_db()
    for name, fn in (("refresh", refresh_sources), ("dropbox", import_dropbox), ("recommendations", generate_recommendations)):
        try:
            log.info("%s: %s", name, json.dumps(fn(db), default=str)[:500])
        except Exception:
            log.exception("job %s failed", name)


def _grading():
    try:
        log.info("grading: %s", grade_and_close(get_db()))
    except Exception:
        log.exception("grading failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.once:
        print(json.dumps(run_all(get_db()), indent=1, default=str))
        return
    from apscheduler.schedulers.blocking import BlockingScheduler
    s = get_settings()
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(_cycle, "interval", minutes=s.refresh_minutes, next_run_time=None, id="cycle")
    sched.add_job(_grading, "interval", minutes=s.grade_minutes, id="grading")
    log.info("scheduler started: refresh every %s min, grading every %s min", s.refresh_minutes, s.grade_minutes)
    _cycle()
    _grading()
    sched.start()


if __name__ == "__main__":
    main()
