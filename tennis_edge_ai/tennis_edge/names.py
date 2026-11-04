"""Player-name normalisation and matching against the players table."""
from __future__ import annotations

import difflib
import re
import unicodedata


def normalize_name(name: str) -> str:
    """Lower-case, strip accents/punctuation, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z ]+", " ", text.replace("-", " ").replace("'", ""))
    return re.sub(r"\s+", " ", text).strip().lower()


def name_variants(name: str) -> set[str]:
    """Variants used for matching: 'First Last', 'Last First', 'F Last'."""
    n = normalize_name(name)
    parts = n.split()
    out = {n}
    if len(parts) >= 2:
        out.add(" ".join(parts[1:] + parts[:1]))
        out.add(f"{parts[0][0]} {' '.join(parts[1:])}")
        out.add(" ".join(parts[-1:] + parts[:-1]))
    return out


class NameMatcher:
    """Resolve free-text player names to player ids for a tour.

    Exact normalised matches win; otherwise an initial+surname or fuzzy match is
    accepted only if it is unambiguous and above ``cutoff``. Ambiguous names return
    None so the caller can flag them instead of guessing.
    """

    def __init__(self, players: list[dict], cutoff: float = 0.88):
        self.cutoff = cutoff
        self.by_norm: dict[str, list[str]] = {}
        self.by_initial: dict[str, list[str]] = {}
        self.recency: dict[str, str] = {}
        for p in players:
            n = p["name_norm"]
            self.by_norm.setdefault(n, []).append(p["player_id"])
            parts = n.split()
            if len(parts) >= 2:
                self.by_initial.setdefault(f"{parts[0][0]} {' '.join(parts[1:])}", []).append(p["player_id"])
                self.by_norm.setdefault(" ".join(parts[1:] + parts[:1]), []).append(p["player_id"])
            if p.get("last_seen"):
                self.recency[p["player_id"]] = p["last_seen"]
        self._keys = list(self.by_norm)

    def _pick(self, ids: list[str]) -> str | None:
        ids = list(dict.fromkeys(ids))
        if len(ids) == 1:
            return ids[0]
        # Prefer the only recently active player when names collide.
        active = [i for i in ids if i in self.recency]
        if len(active) == 1:
            return active[0]
        return None

    def match(self, name: str) -> tuple[str | None, str]:
        n = normalize_name(name)
        if not n:
            return None, "empty name"
        if n in self.by_norm:
            pid = self._pick(self.by_norm[n])
            return (pid, "exact") if pid else (None, "ambiguous exact name")
        parts = n.split()
        if len(parts) >= 2 and len(parts[0]) == 1:
            key = f"{parts[0]} {' '.join(parts[1:])}"
            if key in self.by_initial:
                pid = self._pick(self.by_initial[key])
                return (pid, "initial+surname") if pid else (None, "ambiguous initial+surname")
        close = difflib.get_close_matches(n, self._keys, n=3, cutoff=self.cutoff)
        if len(close) == 1 or (len(close) > 1 and difflib.SequenceMatcher(None, n, close[0]).ratio()
                                - difflib.SequenceMatcher(None, n, close[1]).ratio() > 0.05):
            pid = self._pick(self.by_norm[close[0]])
            if pid:
                return pid, f"fuzzy ({close[0]})"
        return None, "no confident match"
