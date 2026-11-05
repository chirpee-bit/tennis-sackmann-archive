import itertools
import math

import pytest

from tennis_edge.betting.odds import (american_to_decimal, decimal_to_american, devig_two_way, ev_per_unit,
                                      fair_american, implied_prob_american, kelly_fraction, overround,
                                      profit_for_outcome)
from tennis_edge.betting.pickem import (PayoutStructure, breakeven_leg_prob, entry_ev, poisson_binomial,
                                        settle_entry)


def test_conversions_roundtrip():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    for a in (-350, -110, 100, 120, 450):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)
    assert implied_prob_american(-110) == pytest.approx(110 / 210)
    assert fair_american(0.5) == pytest.approx(100)
    with pytest.raises(ValueError):
        american_to_decimal(50)


def test_devig_methods_sum_to_one_and_order():
    for method in ("multiplicative", "power", "shin"):
        a, b = devig_two_way(-150, 130, method)
        assert a + b == pytest.approx(1.0, abs=1e-9)
        assert a > b
    a, b = devig_two_way(-110, -110)
    assert a == pytest.approx(0.5)
    assert overround(-110, -110) == pytest.approx(2 * 110 / 210 - 1)
    # power/shin shift more probability to the favourite than multiplicative for a longshot price
    m = devig_two_way(-800, 550, "multiplicative")[0]
    assert devig_two_way(-800, 550, "power")[0] > m
    assert devig_two_way(-800, 550, "shin")[0] > m


def test_ev_and_kelly():
    assert ev_per_unit(0.5, 100) == pytest.approx(0.0)
    assert ev_per_unit(0.55, -110) == pytest.approx(0.55 * 100 / 110 - 0.45)
    assert kelly_fraction(0.5, -110) == 0.0
    assert kelly_fraction(0.6, 100) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 100, 0.25) == pytest.approx(0.05)


def test_profit_for_outcome():
    assert profit_for_outcome("win", 10, 150) == pytest.approx(15)
    assert profit_for_outcome("loss", 10, 150) == -10
    assert profit_for_outcome("push", 10, 150) == 0
    assert profit_for_outcome("void", 10, None) == 0


def test_poisson_binomial_against_enumeration():
    probs = [0.6, 0.55, 0.7]
    dist = poisson_binomial(probs)
    brute = [0.0] * 4
    for combo in itertools.product([0, 1], repeat=3):
        pr = math.prod(p if c else 1 - p for p, c in zip(probs, combo))
        brute[sum(combo)] += pr
    assert list(dist) == pytest.approx(brute)


POWER2 = PayoutStructure("P2", "PrizePicks", "power2", 2, {2: 3.0}, verified=True, family="P")
FLEX3 = PayoutStructure("F3", "PrizePicks", "flex3", 3, {3: 2.25, 2: 1.25}, verified=True, family="F")


def test_entry_ev_and_breakeven():
    assert entry_ev([0.6, 0.6], POWER2) == pytest.approx(0.36 * 3 - 1)
    assert breakeven_leg_prob(POWER2) == pytest.approx(math.sqrt(1 / 3), abs=1e-9)
    be = breakeven_leg_prob(FLEX3)
    assert entry_ev([be] * 3, FLEX3) == pytest.approx(0, abs=1e-9)


def test_unverified_structure_refuses_ev():
    s = PayoutStructure("X", "PrizePicks", "x", 2, {2: 3.0}, verified=False)
    with pytest.raises(PermissionError):
        entry_ev([0.6, 0.6], s)


def test_settle_entry_with_void_reprices():
    p3 = PayoutStructure("P3", "PrizePicks", "power3", 3, {3: 5.0}, verified=True, family="P")
    structs = {"P2": POWER2, "P3": p3}
    assert settle_entry(["win", "win", "win"], p3, structs)[0] == 5.0
    assert settle_entry(["win", "loss", "win"], p3, structs)[0] == 0.0
    assert settle_entry(["win", "void", "win"], p3, structs)[0] == 3.0      # falls back to 2-leg power
    refund = PayoutStructure("R", "PrizePicks", "r", 2, {2: 3.0}, void_rule="refund", verified=True)
    assert settle_entry(["win", "void"], refund, {})[0] == 1.0
