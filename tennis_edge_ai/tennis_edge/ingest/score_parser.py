"""Parse Sackmann-style score strings into structured outcomes.

Examples handled::

    "7-6(5) 6-4"            completed, 2 sets, 23 games, 1 tiebreak
    "6-3 3-6 7-6(10-8)"     tiebreak score written in full
    "2-6 6-4 [11-9]"        match tiebreak (counted as ONE game, the usual
                            settlement convention for total-games markets)
    "6-4 3-1 RET"           retirement (most props void; status='retired')
    "W/O", "DEF", "Def."    walkover / default
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SET_RE = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\((\d+)(?:-(\d+))?\))?$")
_MTB_RE = re.compile(r"^\[(\d+)-(\d+)\]$")


@dataclass
class SetScore:
    winner_games: int      # games won by the MATCH winner in this set
    loser_games: int
    tiebreak: bool = False
    match_tiebreak: bool = False

    @property
    def games(self) -> int:
        return 1 if self.match_tiebreak else self.winner_games + self.loser_games

    @property
    def complete(self) -> bool:
        if self.match_tiebreak:
            return True
        hi, lo = max(self.winner_games, self.loser_games), min(self.winner_games, self.loser_games)
        if self.tiebreak:
            return True
        return (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo == 6)


@dataclass
class ParsedScore:
    raw: str
    status: str                       # completed / retired / walkover / default / unknown
    sets: list[SetScore] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def total_games(self) -> int | None:
        if self.status in ("walkover", "unknown") or not self.sets:
            return None
        return sum(s.games for s in self.sets)

    @property
    def winner_games(self) -> int | None:
        if self.status in ("walkover", "unknown") or not self.sets:
            return None
        return sum((1 if s.winner_games > s.loser_games else 0) if s.match_tiebreak else s.winner_games
                   for s in self.sets)

    @property
    def loser_games(self) -> int | None:
        if self.status in ("walkover", "unknown") or not self.sets:
            return None
        return sum((1 if s.loser_games > s.winner_games else 0) if s.match_tiebreak else s.loser_games
                   for s in self.sets)

    @property
    def total_sets(self) -> int | None:
        """Number of sets started (equals sets played for completed matches)."""
        if self.status in ("walkover", "unknown") or not self.sets:
            return None
        return len(self.sets)

    @property
    def first_set_games(self) -> int | None:
        if not self.sets or self.status in ("walkover", "unknown"):
            return None
        s = self.sets[0]
        return s.games if s.complete else None

    @property
    def tiebreaks(self) -> int:
        return sum(1 for s in self.sets if s.tiebreak)

    def sets_won(self) -> tuple[int, int]:
        w = sum(1 for s in self.sets if s.complete and s.winner_games > s.loser_games)
        lo = sum(1 for s in self.sets if s.complete and s.loser_games > s.winner_games)
        return w, lo


_RET_TOKENS = {"RET", "RET.", "RETIRED", "ABN", "ABD", "UNFINISHED", "UNF"}
_DEF_TOKENS = {"DEF", "DEF.", "DEFAULT", "DISQ", "DQ"}
_WO_TOKENS = {"W/O", "WO", "W.O.", "WALKOVER"}


def parse_score(score: object) -> ParsedScore:
    if score is None or (isinstance(score, float) and score != score):
        return ParsedScore(raw="", status="unknown")
    raw = str(score).strip()
    if not raw or raw.upper() in {"NA", "NAN", "NONE"}:
        return ParsedScore(raw=raw, status="unknown")
    tokens = raw.replace("  ", " ").split(" ")
    upper = [t.upper() for t in tokens]
    if any(t in _WO_TOKENS for t in upper) or "WALKOVER" in raw.upper():
        return ParsedScore(raw=raw, status="walkover")
    status = "completed"
    if any(t in _RET_TOKENS for t in upper) or "RET" in raw.upper():
        status = "retired"
    elif any(t in _DEF_TOKENS for t in upper):
        status = "default"
    sets: list[SetScore] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        m = _SET_RE.match(t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            tb = m.group(3) is not None or (max(a, b) == 7 and min(a, b) == 6)
            sets.append(SetScore(a, b, tiebreak=tb))
            continue
        m = _MTB_RE.match(t)
        if m:
            sets.append(SetScore(int(m.group(1)), int(m.group(2)), match_tiebreak=True))
            continue
        if t.upper() in _RET_TOKENS | _DEF_TOKENS:
            continue
        # Unrecognised fragment (e.g. 'Played and unfinished'): be conservative.
        return ParsedScore(raw=raw, status="unknown", sets=sets)
    if not sets and status == "completed":
        return ParsedScore(raw=raw, status="unknown")
    if status == "completed" and not all(s.complete for s in sets):
        status = "unknown"
    if status == "completed" and sets:
        w, lo = (sum(1 for s in sets if s.winner_games > s.loser_games),
                 sum(1 for s in sets if s.loser_games > s.winner_games))
        if w <= lo:
            status = "unknown"
    return ParsedScore(raw=raw, status=status, sets=sets)
