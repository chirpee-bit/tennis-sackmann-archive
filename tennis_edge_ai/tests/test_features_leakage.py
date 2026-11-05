"""The feature engine must never use information from the match itself or later matches."""
import numpy as np
import pandas as pd

from tennis_edge.features.builder import orient, run_engine
from tennis_edge.features.engine import FeatureEngine, elo_prob


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    players = [f"ATP:{i}" for i in range(20)]
    skill = rng.normal(0, 1, 20)
    rows = []
    for k in range(n):
        a, b = rng.choice(20, 2, replace=False)
        pa = 1 / (1 + np.exp(-(skill[a] - skill[b])))
        w, l = (a, b) if rng.random() < pa else (b, a)
        rows.append({"match_id": f"m{k}", "est_match_date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=k),
                     "winner_id": players[w], "loser_id": players[l], "surface": "Hard", "status": "completed",
                     "level": "A", "round": "R32", "best_of": 3, "total_games": 22, "total_sets": 2,
                     "first_set_games": 10, "winner_games": 12, "loser_games": 10, "tiebreaks": 0, "minutes": 90,
                     "winner_rank": 10, "loser_rank": 20, "winner_rank_pts": 1000, "loser_rank_pts": 500,
                     "winner_age": 25, "loser_age": 26, "w_ht": 185, "l_ht": 185, "w_hand": "R", "l_hand": "R",
                     "w_ace": rng.integers(0, 15), "w_df": 2, "w_svpt": 70, "w_1stIn": 42, "w_1stWon": 32, "w_2ndWon": 15,
                     "w_SvGms": 11, "w_bpSaved": 2, "w_bpFaced": 3, "l_ace": 4, "l_df": 3, "l_svpt": 72, "l_1stIn": 40,
                     "l_1stWon": 27, "l_2ndWon": 14, "l_SvGms": 11, "l_bpSaved": 4, "l_bpFaced": 7})
    return pd.DataFrame(rows)


def test_features_do_not_depend_on_future_or_current_result():
    m = _synthetic()
    f1, _ = run_engine(m, "ATP", emit_from_year=2015)
    m2 = m.copy()
    k = 250
    # change the result and stats of match k and everything after it
    # swap the players (and their per-player pre-match attributes) -> the result is reversed
    for a, b in (("winner_id", "loser_id"), ("winner_rank", "loser_rank"), ("winner_rank_pts", "loser_rank_pts"),
                 ("winner_age", "loser_age")):
        m2.loc[k:, [a, b]] = m2.loc[k:, [b, a]].to_numpy(copy=True)
    m2.loc[k:, "w_ace"] = 99
    f2, _ = run_engine(m2, "ATP", emit_from_year=2015)
    cols = [c for c in f1.columns if c.startswith(("a_", "b_", "h2h")) and c not in ("a_ht", "b_ht", "a_hand", "b_hand")]
    # rows before k identical
    pd.testing.assert_frame_equal(f1.loc[:k - 1, cols], f2.loc[:k - 1, cols])
    # row k: features identical up to the player swap (pre-match state cannot know the result)
    swap = {c: ("b_" + c[2:] if c.startswith("a_") else "a_" + c[2:] if c.startswith("b_") else c) for c in cols}
    swap["h2h_a"], swap["h2h_b"] = "h2h_b", "h2h_a"
    r1 = f1.loc[k, cols].astype(float)
    r2 = f2.loc[k, [swap[c] for c in cols]].astype(float)
    np.testing.assert_allclose(r1.to_numpy(), r2.to_numpy(), rtol=1e-12, equal_nan=True)


def test_orientation_is_balanced_and_label_consistent():
    m = _synthetic(600)
    f, _ = run_engine(m, "ATP", emit_from_year=2015)
    o = orient(f)
    assert 0.4 < o["y"].mean() < 0.6
    # a_id is the winner exactly when y == 1
    win = dict(zip(m["match_id"], m["winner_id"]))
    assert all((o["a_id"] == o["match_id"].map(win)) == (o["y"] == 1))


def test_elo_updates_zero_sum_for_equal_experience():
    e = FeatureEngine("ATP")
    e.update({"t": 1.0, "winner_id": "A", "loser_id": "B", "status": "completed", "surface": "Hard"})
    assert e.players["A"].elo + e.players["B"].elo == 3000
    assert e.players["A"].elo > 1500
    assert elo_prob(1600, 1400) > 0.75
