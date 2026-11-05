"""Historical match data from Jeff Sackmann's ATP/WTA repositories (local archive).

License: CC BY-NC-SA 4.0 - non-commercial use with attribution. To refresh, pull
the upstream repositories (https://github.com/JeffSackmann/tennis_atp and
tennis_wta) into ``TE_SACKMANN_DIR``; nothing is scraped.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tennis_edge.config import get_settings
from tennis_edge.providers.base import DataProvider, ProviderStatus

MATCH_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date", "match_num",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

_YEAR_RE = re.compile(r"_(\d{4})\.csv$")


class SackmannProvider(DataProvider):
    name = "Sackmann historical archive"
    kind = "historical"
    requires = ["Local copy of tennis_atp / tennis_wta CSV files"]
    terms_note = "CC BY-NC-SA 4.0 - non-commercial use, attribute Jeff Sackmann"

    def __init__(self, base_dir: str | Path | None = None):
        s = get_settings()
        self.base_dir = Path(base_dir) if base_dir else s.sackmann_dir
        self.start_year = s.history_start_year
        self.include_challengers = s.include_challengers
        self.include_itf = s.include_itf
        self.include_futures = s.include_futures

    def status(self) -> ProviderStatus:
        atp, wta = self.base_dir / "atp", self.base_dir / "wta"
        if not atp.is_dir() and not wta.is_dir():
            return self._status(False, f"No atp/ or wta/ folder under {self.base_dir}")
        files = self.match_files("ATP") + self.match_files("WTA")
        if not files:
            return self._status(False, "Folders exist but no match CSV files found")
        last = max(int(_YEAR_RE.search(f.name).group(1)) for f in files)
        return self._status(True, f"{len(files)} match files, latest season file {last}")

    # ------------------------------------------------------------------ files
    def match_files(self, tour: str, start_year: int | None = None) -> list[Path]:
        start = self.start_year if start_year is None else start_year
        folder = self.base_dir / tour.lower()
        if not folder.is_dir():
            return []
        prefix = tour.lower()
        patterns = [(f"{prefix}_matches_[0-9][0-9][0-9][0-9].csv", "main")]
        if tour == "ATP":
            if self.include_challengers:
                patterns.append((f"{prefix}_matches_qual_chall_*.csv", "qual_chall"))
            if self.include_futures:
                patterns.append((f"{prefix}_matches_futures_*.csv", "futures"))
        else:
            if self.include_itf:
                patterns.append((f"{prefix}_matches_qual_itf_*.csv", "qual_itf"))
        out: list[Path] = []
        for pat, _ in patterns:
            for f in sorted(folder.glob(pat)):
                m = _YEAR_RE.search(f.name)
                if m and int(m.group(1)) >= start:
                    out.append(f)
        return out

    def load_matches(self, tour: str, start_year: int | None = None) -> pd.DataFrame:
        frames = []
        for f in self.match_files(tour, start_year):
            df = pd.read_csv(f, low_memory=False, dtype={"tourney_id": str, "tourney_level": str,
                                                          "winner_id": str, "loser_id": str})
            missing = [c for c in MATCH_COLUMNS if c not in df.columns]
            for c in missing:
                df[c] = pd.NA
            df = df[MATCH_COLUMNS].copy()
            df["source_file"] = f.name
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=MATCH_COLUMNS + ["source_file", "tour"])
        out = pd.concat(frames, ignore_index=True)
        out["tour"] = tour
        return out

    def load_players(self, tour: str) -> pd.DataFrame:
        f = self.base_dir / tour.lower() / f"{tour.lower()}_players.csv"
        if not f.exists():
            return pd.DataFrame()
        df = pd.read_csv(f, dtype={"player_id": str}, low_memory=False)
        df["tour"] = tour
        return df

    def load_rankings(self, tour: str, files: tuple[str, ...] = ("20s", "current")) -> pd.DataFrame:
        frames = []
        for suffix in files:
            f = self.base_dir / tour.lower() / f"{tour.lower()}_rankings_{suffix}.csv"
            if f.exists():
                frames.append(pd.read_csv(f, dtype={"player": str}))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True).drop_duplicates(["ranking_date", "player"])
        df["tour"] = tour
        return df
