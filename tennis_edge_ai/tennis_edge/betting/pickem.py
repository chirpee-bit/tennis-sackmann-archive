"""Pick'em (PrizePicks / Tailgate style) payout structures and expected value.

Pick'em entries are NOT priced like sportsbook bets: an entry of n legs pays a
multiplier that depends on how many legs hit. The expected value of an entry
is therefore

    EV = sum_k P(k hits) * multiplier(k) - 1         (per 1 unit staked)

with P(k hits) from the Poisson-binomial distribution of the legs' model
probabilities (independence assumed - correlated legs, e.g. two props from the
same match, violate this and are flagged).

Payout tables must be entered and *verified* by the user (config/payouts.json).
EV is never computed from an unverified table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tennis_edge.config import get_settings


@dataclass(frozen=True)
class PayoutStructure:
    id: str
    platform: str
    label: str
    legs: int
    payouts: dict[int, float]        # hits -> total return multiplier (stake included)
    void_rule: str = "drop_leg"      # drop_leg | refund
    verified: bool = False
    family: str = ""

    def multiplier(self, hits: int) -> float:
        return float(self.payouts.get(hits, 0.0))


def load_payouts(path: Path | None = None) -> dict[str, PayoutStructure]:
    path = path or get_settings().payouts_file
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    out = {}
    for sid, d in raw.get("structures", {}).items():
        legs = int(d["legs"])
        payouts = {int(k): float(v) for k, v in d["payouts"].items()}
        if any(k > legs or k < 0 for k in payouts) or any(v < 0 for v in payouts.values()):
            raise ValueError(f"payout structure {sid} is inconsistent")
        out[sid] = PayoutStructure(sid, d["platform"], d.get("label", sid), legs, payouts,
                                   d.get("void_rule", "drop_leg"), bool(d.get("verified", False)),
                                   d.get("family", sid.rsplit("_", 1)[0]))
    return out


def save_verified_flag(structure_id: str, verified: bool, path: Path | None = None) -> None:
    path = path or get_settings().payouts_file
    raw = json.loads(Path(path).read_text())
    raw["structures"][structure_id]["verified"] = bool(verified)
    Path(path).write_text(json.dumps(raw, indent=2))


def upsert_structure(structure_id: str, platform: str, label: str, legs: int, payouts: dict[int, float],
                     void_rule: str = "drop_leg", verified: bool = False, path: Path | None = None) -> None:
    path = path or get_settings().payouts_file
    raw = json.loads(Path(path).read_text()) if Path(path).exists() else {"structures": {}}
    if any(int(k) > legs for k in payouts):
        raise ValueError("payout hits cannot exceed number of legs")
    raw["structures"][structure_id] = {"platform": platform, "label": label, "legs": int(legs),
                                       "payouts": {str(k): float(v) for k, v in payouts.items()},
                                       "void_rule": void_rule, "verified": bool(verified)}
    Path(path).write_text(json.dumps(raw, indent=2))


def poisson_binomial(probs: list[float] | np.ndarray) -> np.ndarray:
    """P(k successes), k = 0..n, for independent Bernoulli legs."""
    dist = np.array([1.0])
    for p in probs:
        if not 0 <= p <= 1:
            raise ValueError("leg probabilities must be in [0,1]")
        dist = np.convolve(dist, [1 - p, p])
    return dist


def entry_ev(probs: list[float], structure: PayoutStructure, require_verified: bool = True) -> float:
    if require_verified and not structure.verified:
        raise PermissionError(f"payout structure {structure.id} is not verified - EV unavailable")
    if len(probs) != structure.legs:
        raise ValueError(f"structure {structure.id} needs {structure.legs} legs, got {len(probs)}")
    dist = poisson_binomial(probs)
    return float(sum(dist[k] * structure.multiplier(k) for k in range(len(dist))) - 1.0)


def breakeven_leg_prob(structure: PayoutStructure) -> float:
    """Per-leg probability p at which an entry of identical legs has zero EV."""
    lo, hi = 0.0, 1.0

    def ev(p):
        dist = poisson_binomial([p] * structure.legs)
        return sum(dist[k] * structure.multiplier(k) for k in range(len(dist))) - 1.0

    if ev(1.0) <= 0:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        if ev(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def leg_equivalent_ev(p: float, structure: PayoutStructure) -> float:
    """EV of an entry whose legs all had probability p (a per-leg summary, clearly labelled)."""
    return entry_ev([p] * structure.legs, structure)


def settle_entry(leg_outcomes: list[str], structure: PayoutStructure,
                 structures: dict[str, PayoutStructure] | None = None) -> tuple[float, str]:
    """Return (total return multiplier, explanation) for settled legs.

    Void/push legs are removed; the entry is re-priced with the same family's
    structure for the remaining number of legs if one exists, else refunded.
    """
    valid = [o for o in leg_outcomes if o in ("win", "loss")]
    dropped = len(leg_outcomes) - len(valid)
    if any(o not in ("win", "loss", "push", "void") for o in leg_outcomes):
        raise ValueError("unknown leg outcome")
    if dropped == 0:
        hits = sum(o == "win" for o in valid)
        return structure.multiplier(hits), f"{hits}/{structure.legs} hits"
    if structure.void_rule == "refund" or len(valid) < 1:
        return 1.0, f"{dropped} leg(s) void/push -> refund"
    fallback = None
    for s in (structures or {}).values():
        if s.family == structure.family and s.platform == structure.platform and s.legs == len(valid):
            fallback = s
            break
    if fallback is None:
        return 1.0, f"{dropped} leg(s) void/push, no {len(valid)}-leg structure configured -> refund"
    hits = sum(o == "win" for o in valid)
    return fallback.multiplier(hits), f"{dropped} void/push -> repriced as {fallback.id}: {hits}/{fallback.legs} hits"
