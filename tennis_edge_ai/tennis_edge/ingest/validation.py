"""Validation of uploaded / imported projection files (Pydantic)."""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from tennis_edge.markets import load_markets, resolve_market

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 20000

# plausible line ranges per market (outside -> rejected as a data error)
LINE_RANGES = {"total_games": (8, 70), "player_games_won": (0, 40), "first_set_games": (6, 13.5),
               "total_sets": (1.5, 4.5), "aces": (0, 45), "double_faults": (0, 25), "break_points_won": (0, 15),
               "match_winner": (None, None)}

COLUMN_ALIASES = {
    "player": ["player", "player_name", "name", "player name", "selection", "athlete"],
    "opponent": ["opponent", "opp", "vs", "versus", "description", "opponent_name"],
    "market": ["market", "stat", "stat_type", "stat type", "prop", "category"],
    "line": ["line", "line_score", "projection", "value", "points", "total"],
    "over_odds": ["over_odds", "over", "odds_over", "more_odds", "odds", "price"],
    "under_odds": ["under_odds", "under", "odds_under", "less_odds"],
    "start_time": ["start_time", "start", "game_start", "start time", "commence_time", "match_time", "date"],
    "tournament": ["tournament", "event", "league_event", "competition"],
    "surface": ["surface"],
    "tour": ["tour", "league", "circuit"],
    "best_of": ["best_of", "bo", "format"],
    "payout_structure": ["payout_structure", "structure", "entry_type"],
    "platform": ["platform", "site", "book", "bookmaker"],
}


class ProjectionIn(BaseModel):
    platform: str = Field(min_length=2, max_length=40)
    player: str = Field(min_length=2, max_length=80)
    opponent: str | None = Field(default=None, max_length=80)
    market: str
    line: float | None = None
    over_odds: float | None = None
    under_odds: float | None = None
    start_time: datetime
    tournament: str | None = None
    surface: str | None = None
    tour: str | None = None
    best_of: int | None = None
    payout_structure: str | None = None

    @field_validator("player", "opponent")
    @classmethod
    def clean_name(cls, v):
        if v is None:
            return v
        v = re.sub(r"\s+", " ", str(v)).strip()
        if re.search(r"[<>{};=()\[\]|\\]", v) or v.startswith(("'", '"', "@", "+", "-")):
            raise ValueError("name contains forbidden characters")
        return v

    @field_validator("over_odds", "under_odds")
    @classmethod
    def american(cls, v):
        if v is None:
            return v
        if -100 < v < 100 or abs(v) > 100000:
            raise ValueError("odds must be American format (e.g. -110, +150)")
        return v

    @field_validator("tour")
    @classmethod
    def tour_ok(cls, v):
        if v is None:
            return v
        v = v.strip().upper()
        if v in ("ATP", "WTA"):
            return v
        if v in ("CHALLENGER", "ATP CHALLENGER"):
            return "ATP"
        if v in ("WTA 125", "WTA125"):
            return "WTA"
        raise ValueError("tour must be ATP or WTA (other tours are not supported - insufficient data)")

    @field_validator("surface")
    @classmethod
    def surface_ok(cls, v):
        if v is None:
            return v
        s = v.strip().title()
        if s not in ("Hard", "Clay", "Grass", "Carpet"):
            raise ValueError("surface must be Hard/Clay/Grass/Carpet")
        return s

    @field_validator("best_of")
    @classmethod
    def bo_ok(cls, v):
        if v is not None and v not in (3, 5):
            raise ValueError("best_of must be 3 or 5")
        return v

    @field_validator("start_time")
    @classmethod
    def tz(cls, v: datetime):
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    out = {}
    for target, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                out[target] = df[lower[a]]
                break
    return pd.DataFrame(out)


def read_upload(data: bytes, filename: str) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only .csv files are accepted")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large (> {MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
    if b"\x00" in data[:4096]:
        raise ValueError("File looks binary, not CSV")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("File must be UTF-8 encoded") from exc
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if len(df) > MAX_ROWS:
        raise ValueError(f"Too many rows ({len(df)} > {MAX_ROWS})")
    # neutralise spreadsheet formula injection in any text cell
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(r"^[=+\-@](?=[A-Za-z])", "'", regex=True)
    return df


def validate_projection_frame(df: pd.DataFrame, platform: str | None = None
                              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (valid rows, errors). Each error = {row, error, raw}."""
    markets = load_markets()
    mapped = _map_columns(df)
    missing = [c for c in ("player", "market", "start_time") if c not in mapped]
    if missing:
        return [], [{"row": None, "error": f"missing required column(s): {missing}", "raw": list(df.columns)}]
    if platform is None and "platform" not in mapped:
        return [], [{"row": None, "error": "platform not given (choose one or include a 'platform' column)", "raw": None}]
    ok, errors, seen = [], [], set()
    for i, r in mapped.iterrows():
        raw = {k: (None if (v is None or str(v).strip() in ("", "nan", "None")) else str(v).strip())
               for k, v in r.items()}
        rec = dict(raw)
        rec["platform"] = platform or raw.get("platform")
        mkey = resolve_market(raw.get("market") or "", markets)
        if mkey is None:
            errors.append({"row": int(i) + 2, "error": f"unsupported market '{raw.get('market')}'", "raw": raw})
            continue
        rec["market"] = mkey
        for num in ("line", "over_odds", "under_odds"):
            if rec.get(num) is not None:
                try:
                    rec[num] = float(str(rec[num]).replace("+", ""))
                except ValueError:
                    errors.append({"row": int(i) + 2, "error": f"{num} is not numeric", "raw": raw})
                    rec = None
                    break
        if rec is None:
            continue
        if rec.get("best_of") is not None:
            try:
                rec["best_of"] = int(float(rec["best_of"]))
            except ValueError:
                rec["best_of"] = None
        try:
            item = ProjectionIn(**rec)
        except ValidationError as exc:
            errors.append({"row": int(i) + 2, "error": "; ".join(e["msg"] for e in exc.errors()), "raw": raw})
            continue
        if item.market != "match_winner" and item.line is None:
            errors.append({"row": int(i) + 2, "error": "line is required for this market", "raw": raw})
            continue
        lo, hi = LINE_RANGES.get(item.market, (None, None))
        if item.line is not None and lo is not None and not (lo <= item.line <= hi):
            errors.append({"row": int(i) + 2, "error": f"line {item.line} outside plausible range [{lo},{hi}]", "raw": raw})
            continue
        if markets[item.market].player_level is False and not item.opponent:
            errors.append({"row": int(i) + 2, "error": "match-level market needs both players (opponent column)", "raw": raw})
            continue
        key = (item.platform, item.market, item.player.lower(), (item.opponent or "").lower(), item.line,
               item.start_time.isoformat())
        if key in seen:
            errors.append({"row": int(i) + 2, "error": "duplicate row in file", "raw": raw})
            continue
        seen.add(key)
        ok.append(item.model_dump())
    return ok, errors
