"""Odds conversion, margin removal and expected value for sportsbook-style prices."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0 or -100 < a < 100:
        raise ValueError(f"Invalid American odds: {american}")
    return 1 + (a / 100 if a > 0 else 100 / -a)


def decimal_to_american(decimal: float) -> float:
    d = float(decimal)
    if d <= 1:
        raise ValueError(f"Decimal odds must be > 1, got {decimal}")
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


def implied_prob_american(american: float) -> float:
    """Raw implied probability (still includes the bookmaker margin)."""
    return 1 / american_to_decimal(american)


def fair_american(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError("probability must be in (0,1)")
    return decimal_to_american(1 / p)


def overround(*american: float) -> float:
    return sum(implied_prob_american(a) for a in american) - 1


def devig_two_way(odds_a: float, odds_b: float, method: str = "multiplicative") -> tuple[float, float]:
    """Remove the margin from a two-way market (American odds).

    * multiplicative: normalise raw implied probabilities.
    * power: find k with q_a^k + q_b^k = 1 (accounts for favourite-longshot bias).
    * shin: Shin's insider-trading model, closed form for two outcomes.
    """
    qa, qb = implied_prob_american(odds_a), implied_prob_american(odds_b)
    total = qa + qb
    if total <= 0:
        raise ValueError("invalid odds")
    if method == "multiplicative" or total <= 1:
        return qa / total, qb / total
    if method == "power":
        k = brentq(lambda k: qa ** k + qb ** k - 1, 1e-6, 50)
        return qa ** k, qb ** k
    if method == "shin":
        # Two-outcome Shin: solve for z, then p_i = (sqrt(z^2 + 4(1-z) q_i^2 / S) - z) / (2(1-z))
        def ps(z):
            return [(math.sqrt(z * z + 4 * (1 - z) * q * q / total) - z) / (2 * (1 - z)) for q in (qa, qb)]
        z = brentq(lambda z: sum(ps(z)) - 1, 0.0, 0.4)
        pa, pb = ps(z)
        return pa, pb
    raise ValueError(f"unknown devig method {method}")


def ev_per_unit(p_win: float, american: float, p_push: float = 0.0) -> float:
    """Expected profit per 1 unit staked at American odds (push returns stake)."""
    d = american_to_decimal(american)
    p_lose = 1 - p_win - p_push
    return p_win * (d - 1) - p_lose


def kelly_fraction(p_win: float, american: float, fraction: float = 1.0) -> float:
    b = american_to_decimal(american) - 1
    f = (p_win * b - (1 - p_win)) / b
    return max(0.0, f * fraction)


def profit_for_outcome(outcome: str, stake: float, american: float | None) -> float:
    """Realised profit of a single sportsbook-style bet."""
    if outcome in ("push", "void"):
        return 0.0
    if outcome == "loss":
        return -stake
    if outcome == "win":
        if american is None or (isinstance(american, float) and np.isnan(american)):
            raise ValueError("odds required to settle a winning single bet")
        return stake * (american_to_decimal(american) - 1)
    raise ValueError(f"unknown outcome {outcome}")
