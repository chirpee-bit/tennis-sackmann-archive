import numpy as np
import pytest

from tennis_edge.models.markov import (expected_points_per_game, hold_prob, match_distribution, match_win_prob,
                                       prob_over, serve_probs_for_target, set_score_dist, tiebreak_prob)


def test_hold_known_values():
    assert hold_prob(np.array([0.5]))[0] == pytest.approx(0.5)
    assert hold_prob(np.array([0.6]))[0] == pytest.approx(0.7357, abs=1e-4)
    assert hold_prob(np.array([0.7]))[0] == pytest.approx(0.9008, abs=1e-4)


def test_expected_points_per_game_symmetric():
    e = expected_points_per_game(np.array([0.5, 0.6, 0.4]))
    assert e[0] == pytest.approx(6.75, abs=1e-6)
    assert e[1] == pytest.approx(e[2])


def test_tiebreak_symmetry_and_edge():
    assert tiebreak_prob(np.array([0.62]), np.array([0.62]))[0] == pytest.approx(0.5, abs=1e-9)
    assert tiebreak_prob(np.array([0.7]), np.array([0.6]))[0] > 0.5


def test_set_dist_sums_to_one():
    d = set_score_dist(np.array([0.65, 0.55]), np.array([0.6, 0.7]), True)
    total = sum(d.values())
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


@pytest.mark.parametrize("bo", [3, 5])
def test_match_distribution_consistency(bo):
    pa, pb = np.array([0.64, 0.6, 0.7]), np.array([0.6, 0.6, 0.55])
    md = match_distribution(pa, pb, bo)
    np.testing.assert_allclose(md.joint.sum(axis=(1, 2)), 1.0, atol=1e-10)
    np.testing.assert_allclose(md.sets_pmf.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(md.first_set_pmf.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(md.p_match, match_win_prob(pa, pb, bo), atol=1e-10)
    assert md.p_match[1] == pytest.approx(0.5, abs=1e-12)        # equal players
    assert md.p_match[0] > 0.5 and md.p_match[2] > md.p_match[0]
    need = 3 if bo == 5 else 2
    assert md.sets_pmf[:, :need].sum() == pytest.approx(0)


def test_markov_matches_monte_carlo():
    rng = np.random.default_rng(1)
    pa, pb, n = 0.66, 0.61, 20000

    def play_game(p):
        a = b = 0
        while True:
            if rng.random() < p:
                a += 1
            else:
                b += 1
            if a >= 4 and a - b >= 2:
                return True
            if b >= 4 and b - a >= 2:
                return False

    def play_tb(first_a):
        a = b = 0
        k = 0
        while True:
            a_serving = (((k + 1) // 2) % 2 == 0) == first_a
            if a_serving:
                win = rng.random() < pa
            else:
                win = rng.random() >= pb
            a, b = a + win, b + (not win)
            k += 1
            if (a >= 7 or b >= 7) and abs(a - b) >= 2:
                return a > b

    totals, wins = [], 0
    for _ in range(n):
        a_first = rng.random() < 0.5
        sa = sb = ga_tot = gb_tot = 0
        while sa < 2 and sb < 2:
            ga = gb = 0
            k = 0
            while True:
                if ga == 6 and gb == 6:
                    if play_tb(((k % 2) == 0) == a_first):
                        ga += 1
                    else:
                        gb += 1
                    k += 1
                    break
                a_serv = ((k % 2) == 0) == a_first
                if a_serv:
                    if play_game(pa):
                        ga += 1
                    else:
                        gb += 1
                else:
                    if play_game(pb):
                        gb += 1
                    else:
                        ga += 1
                k += 1
                if (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2:
                    break
            if k % 2 == 1:
                a_first = not a_first
            ga_tot += ga
            gb_tot += gb
            sa, sb = sa + (ga > gb), sb + (gb > ga)
        totals.append(ga_tot + gb_tot)
        wins += sa == 2
    md = match_distribution(np.array([pa]), np.array([pb]), 3)
    mean_dp = (md.total_games_pmf[0] * np.arange(md.total_games_pmf.shape[1])).sum()
    assert wins / n == pytest.approx(md.p_match[0], abs=0.012)
    assert np.mean(totals) == pytest.approx(mean_dp, abs=0.15)
    over, _, _ = prob_over(md.total_games_pmf, 22.5)
    assert np.mean(np.array(totals) > 22.5) == pytest.approx(over[0], abs=0.012)


def test_serve_probs_for_target_recovers_probability():
    target = np.array([0.3, 0.5, 0.8])
    pa, pb = serve_probs_for_target(np.array([1.24, 1.24, 1.3]), target, np.array([3, 5, 3]))
    np.testing.assert_allclose(match_win_prob(pa, pb, np.array([3, 5, 3])), target, atol=1e-4)
    np.testing.assert_allclose(pa + pb, [1.24, 1.24, 1.3], atol=1e-9)


def test_prob_over_push():
    pmf = np.zeros((1, 30))
    pmf[0, [20, 22, 24]] = [0.25, 0.5, 0.25]
    o, u, p = prob_over(pmf, 22)
    assert (o[0], u[0], p[0]) == (0.25, 0.25, 0.5)
