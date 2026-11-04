import pytest

from tennis_edge.ingest.score_parser import parse_score


@pytest.mark.parametrize("score,status,games,sets,first,wg,lg,tb", [
    ("7-6(5) 6-4", "completed", 23, 2, 13, 13, 10, 1),
    ("6-2 6-1", "completed", 15, 2, 8, 12, 3, 0),
    ("3-6 7-6(10-8) 6-4", "completed", 32, 3, 9, 16, 16, 1),
    ("2-6 6-4 [11-9]", "completed", 19, 3, 8, 9, 10, 0),   # match TB counts as one game
    ("6-4 3-1 RET", "retired", 14, 2, 10, 9, 5, 0),
    ("6-7(3) 7-6(5) 6-7(4) 7-6(2) 7-6(8)", "completed", 65, 5, 13, 33, 32, 5),
    ("6-4 6-7(5) 3-6 6-3 70-68", "completed", 179, 5, 10, 91, 88, 1),
])
def test_parse_completed_and_retired(score, status, games, sets, first, wg, lg, tb):
    p = parse_score(score)
    assert p.status == status
    assert p.total_games == games
    assert p.total_sets == sets
    assert p.first_set_games == first
    assert p.winner_games == wg
    assert p.loser_games == lg
    assert p.tiebreaks == tb


def test_walkover_and_unknown():
    assert parse_score("W/O").status == "walkover"
    assert parse_score("W/O").total_games is None
    assert parse_score(None).status == "unknown"
    assert parse_score("&nbsp;").status == "unknown"
    assert parse_score("DEF").status == "default"


def test_incomplete_without_marker_is_unknown():
    # "6-2 0-0" with no RET is not a valid completed match -> excluded from targets
    assert parse_score("6-2 0-0").status == "unknown"
    # Fast4 / short sets are not standard and are excluded
    assert parse_score("4-1 4-0 4-0").status == "unknown"


def test_first_set_incomplete_on_retirement():
    p = parse_score("3-1 RET")
    assert p.status == "retired"
    assert p.first_set_games is None
