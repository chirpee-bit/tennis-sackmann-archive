"""Exact Markov-chain model of a tennis match (vectorised over matches).

Inputs are, for each match, ``pa`` = P(A wins a point on A's serve) and
``pb`` = P(B wins a point on B's serve). Points are assumed i.i.d. given these
probabilities (the classic Barnett-Clarke / Newton-Keller model). Everything is
computed by exact dynamic programming - no Monte-Carlo noise:

* game hold probability (closed form)
* tiebreak win probability (point DP with closed-form deuce)
* set-score distribution conditional on who serves first
* the joint distribution of (games won by A, games won by B) over the whole match
  for best-of-3 or best-of-5, accounting for the serve order carrying across
  sets (a set with an odd number of games swaps the first server)

From the joint distribution we read total games, each player's games won,
number of sets, first-set games and match-win probability.

Conventions: a set tiebreak counts as one game (7-6). Final sets are played
with a tiebreak at 6-6 (all four Slams since 2022; the 10-point final-set
tiebreak is approximated by the 7-point one).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


def hold_prob(p: np.ndarray) -> np.ndarray:
    """P(server holds) given P(server wins a point on serve)."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    q = 1 - p
    pre = p ** 4 * (1 + 4 * q + 10 * q ** 2)
    deuce = 20 * p ** 3 * q ** 3 * (p ** 2 / (1 - 2 * p * q))
    return pre + deuce


def expected_points_per_game(p: np.ndarray) -> np.ndarray:
    """Expected number of points in a service game."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    q = 1 - p
    # probabilities of finishing at 4, 5, 6 points or reaching deuce (6 points played)
    e4 = p ** 4 + q ** 4
    e5 = 4 * (p ** 4 * q + q ** 4 * p)
    e6 = 10 * (p ** 4 * q ** 2 + q ** 4 * p ** 2)
    d = 20 * p ** 3 * q ** 3
    from_deuce = 2.0 / (p ** 2 + q ** 2)
    return 4 * e4 + 5 * e5 + 6 * e6 + d * (6 + from_deuce)


def tiebreak_prob(pf: np.ndarray, po: np.ndarray, target: int = 7) -> np.ndarray:
    """P(first server wins the tiebreak). pf/po = serve-point win probs of first/other server."""
    pf = np.clip(np.asarray(pf, dtype=float), _EPS, 1 - _EPS)
    po = np.clip(np.asarray(po, dtype=float), _EPS, 1 - _EPS)
    n = target - 1
    P = {(0, 0): np.ones_like(pf)}
    win = np.zeros_like(pf)
    for total in range(0, 2 * n + 1):
        for i in range(max(0, total - n), min(total, n) + 1):
            j = total - i
            if (i, j) not in P:
                continue
            pr = P.pop((i, j))
            if i == n and j == n:
                num = pf * (1 - po)
                win += pr * num / (num + (1 - pf) * po)
                continue
            k = i + j
            first_serving = ((k + 1) // 2) % 2 == 0
            pw = pf if first_serving else 1 - po      # P(first server wins this point)
            for (ni, nj, pp) in ((i + 1, j, pw), (i, j + 1, 1 - pw)):
                if ni == target:                        # first server reaches target (opponent < n here)
                    win += pr * pp
                elif nj == target:
                    pass
                else:
                    P[(ni, nj)] = P.get((ni, nj), 0) + pr * pp
    return win


def set_score_dist(pa: np.ndarray, pb: np.ndarray, a_serves_first: bool, tb_target: int = 7
                   ) -> dict[tuple[int, int], np.ndarray]:
    """Distribution of final set scores (games A, games B). Values are arrays over matches."""
    ha, hb = hold_prob(pa), hold_prob(pb)
    P = {(0, 0): np.ones_like(ha)}
    out: dict[tuple[int, int], np.ndarray] = {}
    for total in range(0, 13):
        for i in range(0, total + 1):
            j = total - i
            if (i, j) not in P:
                continue
            pr = P.pop((i, j))
            if i == 6 and j == 6:
                if a_serves_first:
                    pw = tiebreak_prob(pa, pb, tb_target)
                else:
                    pw = 1 - tiebreak_prob(pb, pa, tb_target)
                out[(7, 6)] = out.get((7, 6), 0) + pr * pw
                out[(6, 7)] = out.get((6, 7), 0) + pr * (1 - pw)
                continue
            a_serving = ((i + j) % 2 == 0) == a_serves_first
            pwin = ha if a_serving else 1 - hb
            for ni, nj, pp in ((i + 1, j, pwin), (i, j + 1, 1 - pwin)):
                done = (ni >= 6 and ni - nj >= 2) or (nj >= 6 and nj - ni >= 2)
                if done:
                    out[(ni, nj)] = out.get((ni, nj), 0) + pr * pp
                else:
                    P[(ni, nj)] = P.get((ni, nj), 0) + pr * pp
    return out


@dataclass
class MatchDistribution:
    p_match: np.ndarray            # (N,) P(A wins match)
    joint: np.ndarray              # (N, G, G) P(A games = i, B games = j)
    sets_pmf: np.ndarray           # (N, 6) P(total sets = k)
    first_set_pmf: np.ndarray      # (N, 14) P(first-set games = k)
    p_first_set: np.ndarray        # (N,) P(A wins first set)

    @property
    def total_games_pmf(self) -> np.ndarray:
        n, g, _ = self.joint.shape
        tot = np.zeros((n, 2 * g))
        for i in range(g):
            tot[:, i:i + g] += self.joint[:, i, :]
        return tot

    @property
    def a_games_pmf(self) -> np.ndarray:
        return self.joint.sum(axis=2)

    @property
    def b_games_pmf(self) -> np.ndarray:
        return self.joint.sum(axis=1)


def _pmf_mean(pmf: np.ndarray) -> np.ndarray:
    return (pmf * np.arange(pmf.shape[1])).sum(axis=1)


def match_distribution(pa: np.ndarray, pb: np.ndarray, best_of: int) -> MatchDistribution:
    """Full outcome distribution; serve order of the first set is a coin toss."""
    pa = np.atleast_1d(np.asarray(pa, dtype=float))
    pb = np.atleast_1d(np.asarray(pb, dtype=float))
    n = len(pa)
    need = 3 if best_of == 5 else 2
    g = 7 * (2 * need - 1) + 1
    sets = {True: set_score_dist(pa, pb, True), False: set_score_dist(pa, pb, False)}
    # state: (sets A, sets B, A serves first in next set) -> joint (N,G,G)
    states: dict[tuple[int, int, bool], np.ndarray] = {}
    init = np.zeros((n, g, g))
    init[:, 0, 0] = 0.5
    states[(0, 0, True)] = init.copy()
    states[(0, 0, False)] = init.copy()
    final = np.zeros((n, g, g))
    sets_pmf = np.zeros((n, 6))
    p_match = np.zeros(n)
    for _ in range(2 * need - 1):
        new: dict[tuple[int, int, bool], np.ndarray] = {}
        for (sa, sb, afirst), arr in states.items():
            for (x, y), pr in sets[afirst].items():
                nsa, nsb = sa + (x > y), sb + (y > x)
                nfirst = afirst if (x + y) % 2 == 0 else (not afirst)
                shifted = np.zeros_like(arr)
                shifted[:, x:, y:] = arr[:, :g - x, :g - y] * pr[:, None, None]
                if nsa == need or nsb == need:
                    final += shifted
                    mass = shifted.sum(axis=(1, 2))
                    sets_pmf[:, nsa + nsb] += mass
                    if nsa == need:
                        p_match += mass
                else:
                    key = (nsa, nsb, nfirst)
                    new[key] = new[key] + shifted if key in new else shifted
        states = new
        if not states:
            break
    first = np.zeros((n, 14))
    p_first = np.zeros(n)
    for afirst in (True, False):
        for (x, y), pr in sets[afirst].items():
            first[:, x + y] += 0.5 * pr
            if x > y:
                p_first += 0.5 * pr
    return MatchDistribution(p_match, final, sets_pmf, first, p_first)


def match_win_prob(pa: np.ndarray, pb: np.ndarray, best_of: np.ndarray | int) -> np.ndarray:
    """Fast P(A wins match) (no game distribution). ``best_of`` may vary per match."""
    pa = np.atleast_1d(np.asarray(pa, dtype=float))
    pb = np.atleast_1d(np.asarray(pb, dtype=float))
    bo = np.broadcast_to(np.asarray(best_of), pa.shape)
    res = np.zeros_like(pa)
    sets = {True: set_score_dist(pa, pb, True), False: set_score_dist(pa, pb, False)}
    # per first-server: P(A wins set), P(set has odd games)
    win = {f: sum(pr for (x, y), pr in d.items() if x > y) for f, d in sets.items()}
    odd_w = {f: sum(pr for (x, y), pr in d.items() if x > y and (x + y) % 2) for f, d in sets.items()}
    odd_l = {f: sum(pr for (x, y), pr in d.items() if y > x and (x + y) % 2) for f, d in sets.items()}
    for best in (3, 5):
        mask = bo == best
        if not mask.any():
            continue
        need = 3 if best == 5 else 2
        st = {(0, 0, True): np.full(pa.shape, 0.5), (0, 0, False): np.full(pa.shape, 0.5)}
        total = np.zeros_like(pa)
        for _ in range(2 * need - 1):
            new = {}
            for (sa, sb, f), pr in st.items():
                for a_won, p_set, p_odd in ((True, win[f], odd_w[f]), (False, 1 - win[f], odd_l[f])):
                    nsa, nsb = sa + a_won, sb + (not a_won)
                    for nf, pp in ((not f, p_odd), (f, p_set - p_odd)):
                        m = pr * pp
                        if nsa == need:
                            total += m
                        elif nsb == need:
                            pass
                        else:
                            new[(nsa, nsb, nf)] = new.get((nsa, nsb, nf), 0) + m
            st = new
        res[mask] = total[mask]
    return res


def serve_probs_for_target(p_sum: np.ndarray, p_target: np.ndarray, best_of: np.ndarray,
                           iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Find (pa, pb) with pa + pb = p_sum whose match-win probability equals ``p_target``.

    Keeps the overall serve dominance (which drives game totals) from the serve
    statistics while making the Markov model agree with the calibrated winner
    model. Solved by vectorised bisection on d = pa - pb.
    """
    p_sum = np.asarray(p_sum, dtype=float)
    target = np.clip(np.asarray(p_target, dtype=float), 1e-4, 1 - 1e-4)
    lo = np.full(p_sum.shape, -0.6)
    hi = np.full(p_sum.shape, 0.6)
    for _ in range(iters):
        mid = (lo + hi) / 2
        pa = np.clip((p_sum + mid) / 2, 0.2, 0.98)
        pb = np.clip((p_sum - mid) / 2, 0.2, 0.98)
        pm = match_win_prob(pa, pb, best_of)
        lo = np.where(pm < target, mid, lo)
        hi = np.where(pm >= target, mid, hi)
    d = (lo + hi) / 2
    return np.clip((p_sum + d) / 2, 0.2, 0.98), np.clip((p_sum - d) / 2, 0.2, 0.98)


def prob_over(pmf: np.ndarray, line: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P(over), P(under), P(push)) for a line given a pmf over 0..K (rows = matches)."""
    pmf = np.atleast_2d(pmf)
    k = np.arange(pmf.shape[1])
    line = np.broadcast_to(np.asarray(line, dtype=float), (pmf.shape[0],))[:, None]
    over = (pmf * (k[None, :] > line)).sum(axis=1)
    under = (pmf * (k[None, :] < line)).sum(axis=1)
    push = (pmf * (k[None, :] == line)).sum(axis=1)
    return over, under, push
