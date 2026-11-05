"""Tennis domain constants: round ordering, estimated match dates, level groups."""
from __future__ import annotations

import pandas as pd

ROUND_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "ER": 5, "R128": 6, "R64": 7, "R32": 8,
               "R16": 9, "RR": 9, "QF": 10, "SF": 11, "BR": 12, "F": 13}

# Day offset from the tournament start date (Sackmann's tourney_date) per round.
# Sackmann files have no per-match dates, so these are ESTIMATES used only for
# rest-day / time-decay features. Ordering within a tournament uses ROUND_ORDER.
_OFFSETS_SLAM = {"Q1": -6, "Q2": -5, "Q3": -4, "Q4": -3, "R128": 1, "R64": 3, "R32": 5, "R16": 7,
                 "QF": 9, "SF": 11, "BR": 12, "F": 13}
_OFFSETS_BIG = {"Q1": -2, "Q2": -1, "Q3": -1, "R128": 1, "R64": 2, "R32": 4, "R16": 6,
                "QF": 8, "SF": 9, "BR": 10, "F": 10}
_OFFSETS_WEEK = {"Q1": -2, "Q2": -1, "Q3": 0, "Q4": 0, "ER": 0, "R128": 1, "R64": 1, "R32": 1,
                 "R16": 3, "RR": 3, "QF": 4, "SF": 5, "BR": 6, "F": 6}

SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def normalize_surface(surface: object) -> str:
    if not isinstance(surface, str) or not surface.strip():
        return "Hard"
    s = surface.strip().title()
    return s if s in SURFACES else "Hard"


def elo_surface(surface: str) -> str:
    """Carpet is rare and indoor-fast; pool it with Hard for surface ratings."""
    return "Hard" if surface == "Carpet" else surface


def level_group(tour: str, level: object, round_: object = None) -> str:
    lv = str(level) if level is not None and level == level else ""
    if lv == "G":
        return "Grand Slam"
    if tour == "ATP":
        return {"M": "Masters 1000", "A": "ATP 250/500", "F": "Tour Finals", "D": "Team (Davis Cup)",
                "O": "Olympics", "C": "Challenger", "S": "Futures/ITF", "15": "Futures/ITF",
                "25": "Futures/ITF"}.get(lv, "Other")
    return {"PM": "WTA 1000", "P": "WTA 500", "I": "WTA 250/500", "T1": "WTA 1000", "T2": "WTA 500",
            "T3": "WTA 250/500", "T4": "WTA 250/500", "T5": "WTA 250/500", "F": "Tour Finals",
            "D": "Team (BJK Cup)", "O": "Olympics", "W": "WTA 250/500", "C": "WTA 125",
            "CC": "Other"}.get(lv, "Futures/ITF" if lv.isdigit() or lv.endswith("+H") else "Other")


def estimate_match_dates(tourney_date: pd.Series, round_: pd.Series, level: pd.Series,
                         best_of: pd.Series, draw_size: pd.Series) -> pd.Series:
    base = pd.to_datetime(tourney_date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    slam = (level.astype(str) == "G") & (best_of.fillna(3).astype(float) >= 5) | (level.astype(str) == "G")
    big = (~slam) & (draw_size.fillna(32).astype(float) >= 96)
    r = round_.astype(str)
    off = r.map(_OFFSETS_WEEK).fillna(2)
    off = off.where(~big, r.map(_OFFSETS_BIG).fillna(4))
    off = off.where(~slam, r.map(_OFFSETS_SLAM).fillna(6))
    return base + pd.to_timedelta(off, unit="D")
