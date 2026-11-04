import pandas as pd

from tennis_edge.ingest.historical import ingest_tour, normalize_matches
from tennis_edge.providers.sackmann import MATCH_COLUMNS, SackmannProvider

HEADER = ",".join(MATCH_COLUMNS)


def _row(**kw):
    base = {c: "" for c in MATCH_COLUMNS}
    base.update(dict(tourney_id="2024-1", tourney_name="Test Open", surface="Hard", draw_size=32,
                     tourney_level="A", tourney_date=20240101, best_of=3))
    base.update(kw)
    return ",".join(str(base[c]) for c in MATCH_COLUMNS)


def _write(tmp_path):
    atp = tmp_path / "atp"
    atp.mkdir()
    rows = [
        _row(match_num=1, winner_id=1, winner_name="Ann Ace", loser_id=2, loser_name="Bob Base", score="6-4 6-4", round="R32",
             w_ace=5, w_df=1, w_svpt=60, w_1stIn=40, w_1stWon=30, w_2ndWon=10, w_SvGms=10, w_bpSaved=1, w_bpFaced=2,
             l_ace=2, l_df=3, l_svpt=62, l_1stIn=35, l_1stWon=22, l_2ndWon=12, l_SvGms=10, l_bpSaved=2, l_bpFaced=4),
        # exact duplicate of match 1 under a different match_num -> must be dropped
        _row(match_num=99, winner_id=1, winner_name="Ann Ace", loser_id=2, loser_name="Bob Base", score="6-4 6-4", round="R32"),
        _row(match_num=2, winner_id=1, winner_name="Ann Ace", loser_id=3, loser_name="Cy Clay", score="6-3 3-1 RET", round="R16"),
        _row(match_num=3, winner_id=4, winner_name="Di Drop", loser_id=4, loser_name="Di Drop", score="6-0 6-0", round="R16"),
    ]
    (atp / "atp_matches_2024.csv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    (atp / "atp_players.csv").write_text("player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n"
                                         "1,Ann,Ace,R,20000101,USA,185,\n2,Bob,Base,L,19990101,FRA,180,\n")
    return tmp_path


def test_ingest_idempotent_and_dedup(tmp_db, tmp_path, monkeypatch):
    base = _write(tmp_path)
    monkeypatch.setenv("TE_HISTORY_START_YEAR", "2020")
    prov = SackmannProvider(base)
    rep = ingest_tour(tmp_db, "ATP", prov)
    assert rep.duplicates_dropped == 1
    assert rep.invalid_dropped == 1          # self-match removed
    assert rep.matches_inserted == 2
    assert rep.stats_inserted == 2
    again = ingest_tour(tmp_db, "ATP", prov)
    assert again.matches_inserted == 0       # idempotent
    m = tmp_db.df("SELECT * FROM matches ORDER BY round_order")
    assert list(m["status"]) == ["completed", "retired"]
    assert m.iloc[0]["total_games"] == 20
    # chronological ordering: R16 estimated after R32
    assert m.iloc[1]["est_match_date"] > m.iloc[0]["est_match_date"]


def test_normalize_orders_rounds():
    df = pd.DataFrame([{c: None for c in MATCH_COLUMNS} | dict(tourney_id="x", tourney_date=20240101, winner_id="1",
                       loser_id="2", round=r, match_num=i, score="6-1 6-1", best_of=3, tourney_level="A", draw_size=32,
                       surface="Clay") for i, r in enumerate(["F", "QF", "R32"])])
    out, _, _ = normalize_matches(df.assign(tour="ATP", source_file="f"), "ATP")
    # same players, distinct rounds -> not duplicates; sorted R32 < QF < F
    assert list(out["round"]) == ["R32", "QF", "F"]
