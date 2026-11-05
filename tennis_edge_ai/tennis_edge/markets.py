"""Market registry loaded from ``config/markets.json``.

The registry describes markets the *model* can price. Whether a platform actually
offers a market is a separate, user-verified flag - never assumed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tennis_edge.config import get_settings


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    kind: str              # binary | count
    player_level: bool     # True if the projection is about one player
    aliases: tuple[str, ...]
    platforms: dict[str, bool]

    def verified_for(self, platform: str) -> bool:
        return bool(self.platforms.get(platform, False))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_markets(path: Path | None = None) -> dict[str, Market]:
    path = path or get_settings().markets_file
    raw = json.loads(Path(path).read_text())
    out: dict[str, Market] = {}
    for key, m in raw["markets"].items():
        out[key] = Market(key=key, label=m["label"], kind=m["kind"], player_level=bool(m["player_level"]),
                          aliases=tuple(m.get("aliases", [])), platforms=dict(m.get("platforms", {})))
    return out


def resolve_market(name: str, markets: dict[str, Market] | None = None) -> str | None:
    """Map a platform's market label (e.g. 'Games Won') to an internal key, or None."""
    markets = markets or load_markets()
    n = _norm(name)
    if n.replace(" ", "_") in markets:
        return n.replace(" ", "_")
    for key, m in markets.items():
        if n == _norm(m.label) or n in {_norm(a) for a in m.aliases}:
            return key
    return None


def set_platform_verified(market_key: str, platform: str, verified: bool, path: Path | None = None) -> None:
    path = path or get_settings().markets_file
    raw = json.loads(Path(path).read_text())
    if market_key not in raw["markets"]:
        raise KeyError(market_key)
    raw["markets"][market_key].setdefault("platforms", {})[platform] = bool(verified)
    Path(path).write_text(json.dumps(raw, indent=2))
