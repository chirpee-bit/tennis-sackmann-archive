"""Write a clearly-labelled SIMULATED projection CSV for demoing the pipeline.

The players are real (from the historical database) but every LINE IS FAKE -
generated from rough tour averages plus noise. The tournament column says
"SIMULATED DEMO - NOT REAL LINES" and the file name starts with
prizepicks_SIMULATED_. Do not treat any output based on it as a real opportunity.

    python scripts/make_simulated_projections.py [--pairs 8] [--out var/imports]
"""
import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from tennis_edge.config import get_settings
from tennis_edge.db.database import get_db

LINES = {  # (ATP, WTA) rough centre values for best-of-3
    "Total Games": (22.5, 21.5), "Games Won": (11.5, 10.5), "Aces": (6.5, 2.5),
    "Double Faults": (2.5, 3.5), "Break Points Won": (2.5, 3.5), "1st Set Games": (9.5, 9.5),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--out", default=str(get_settings().import_dropbox_dir))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    db = get_db()
    rows = []
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for tour, ti in (("ATP", 0), ("WTA", 1)):
        top = db.query("SELECT p.full_name FROM rankings r JOIN players p ON p.player_id=r.player_id WHERE r.tour=? AND "
                       "r.ranking_date=(SELECT MAX(ranking_date) FROM rankings WHERE tour=?) AND r.rank<=40 ORDER BY r.rank",
                       (tour, tour))
        names = [t["full_name"] for t in top]
        rng.shuffle(names)
        for i in range(min(a.pairs, len(names) // 2)):
            p, o = names[2 * i], names[2 * i + 1]
            start = now + timedelta(days=1 + i % 3, hours=2 * i)
            for market, centres in LINES.items():
                line = centres[ti] + rng.choice([-1, 0, 0, 1])
                rows.append({"platform": "PrizePicks", "player": p, "opponent": o, "market": market, "line": line,
                             "start_time": start.isoformat(), "tournament": "SIMULATED DEMO - NOT REAL LINES",
                             "surface": "Hard", "tour": tour, "best_of": 3, "payout_structure": "PP_POWER_2"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"prizepicks_SIMULATED_{now.strftime('%Y%m%dT%H%M')}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {len(rows)} SIMULATED rows -> {path}")


if __name__ == "__main__":
    main()
