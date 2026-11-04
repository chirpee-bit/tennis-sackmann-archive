import json

import httpx
import pandas as pd
import pytest

from tennis_edge.ingest.validation import read_upload, validate_projection_frame
from tennis_edge.markets import resolve_market
from tennis_edge.names import NameMatcher, normalize_name
from tennis_edge.providers.licensed_feed import LicensedFeedProvider
from tennis_edge.providers.odds_api import OddsAPIProvider
from tennis_edge.providers.base import ProviderUnavailable


def test_market_aliases():
    assert resolve_market("Total Games") == "total_games"
    assert resolve_market("Games Won") == "player_games_won"
    assert resolve_market("1st Set Games") == "first_set_games"
    assert resolve_market("Fantasy Score") is None


def test_name_matching():
    assert normalize_name("Félix Auger-Aliassime") == "felix auger aliassime"
    m = NameMatcher([{"player_id": "ATP:1", "name_norm": "jannik sinner", "last_seen": "2026"},
                     {"player_id": "ATP:2", "name_norm": "carlos alcaraz", "last_seen": "2026"}])
    assert m.match("Sinner Jannik")[0] == "ATP:1"
    assert m.match("J. Sinner")[0] == "ATP:1"
    assert m.match("Carlos Alcaras")[0] == "ATP:2"
    assert m.match("Roger Federer")[0] is None


def test_upload_validation_rules():
    csv = ("player,opponent,stat_type,line,start_time,tour,over_odds\n"
           "Jannik Sinner,Carlos Alcaraz,Total Games,38.5,2030-01-01T10:00:00Z,ATP,\n"
           "Jannik Sinner,Carlos Alcaraz,Total Games,38.5,2030-01-01T10:00:00Z,ATP,\n"      # duplicate
           "Jannik Sinner,,Total Games,22.5,2030-01-01T10:00:00Z,ATP,\n"                    # match market w/o opponent
           "Jannik Sinner,Carlos Alcaraz,Aces,400,2030-01-01T10:00:00Z,ATP,\n"              # implausible line
           "Jannik Sinner,Carlos Alcaraz,Fantasy Score,40,2030-01-01T10:00:00Z,ATP,\n"      # unsupported
           "Jannik Sinner,Carlos Alcaraz,Aces,9.5,2030-01-01T10:00:00Z,ITF,\n"              # bad tour
           "Jannik Sinner,Carlos Alcaraz,Aces,9.5,2030-01-01T10:00:00Z,ATP,50\n"            # bad odds
           "=cmd(),Carlos Alcaraz,Aces,9.5,2030-01-01T10:00:00Z,ATP,\n")                     # formula injection
    df = read_upload(csv.encode(), "x.csv")
    ok, errors = validate_projection_frame(df, "PrizePicks")
    assert len(ok) == 1
    msgs = " | ".join(e["error"] for e in errors)
    for frag in ("duplicate", "needs both players", "outside plausible", "unsupported market", "tour must be",
                 "American", "forbidden"):
        assert frag in msgs
    with pytest.raises(ValueError):
        read_upload(b"a,b\n1,2", "x.xlsx")
    with pytest.raises(ValueError):
        read_upload(b"\x00\x01binary", "x.csv")


def test_odds_api_unavailable_without_key():
    p = OddsAPIProvider(api_key="")
    assert p.status().available is False
    with pytest.raises(ProviderUnavailable):
        p.tennis_sports()


def test_odds_api_parsing_with_mock_transport():
    events = [{"home_team": "Jannik Sinner", "away_team": "Carlos Alcaraz", "commence_time": "2030-01-01T10:00:00Z",
               "sport_title": "ATP Finals", "bookmakers": [{"key": "bk", "title": "BookA", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Jannik Sinner", "price": -150}, {"name": "Carlos Alcaraz", "price": 130}]},
                   {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 22.5},
                                                  {"name": "Under", "price": -110, "point": 22.5}]}]}]}]

    def handler(request: httpx.Request):
        assert request.url.params["apiKey"] == "k"
        if request.url.path.endswith("/sports"):
            return httpx.Response(200, json=[{"key": "tennis_atp_finals", "active": True}, {"key": "soccer_x", "active": True}],
                                  headers={"x-requests-remaining": "499"})
        return httpx.Response(200, json=events, headers={"x-requests-remaining": "498"})

    p = OddsAPIProvider(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert [s["key"] for s in p.tennis_sports()] == ["tennis_atp_finals"]
    rows = p.parse_events(events, "tennis_atp_finals")
    assert {r["market"] for r in rows} == {"match_winner", "total_games"}
    tot = next(r for r in rows if r["market"] == "total_games")
    assert tot["line"] == 22.5 and tot["over_odds"] == -110


def test_licensed_feed_requires_configuration():
    p = LicensedFeedProvider("PrizePicks", url="", key="")
    p.url = p.key = None
    assert p.status().available is False
    with pytest.raises(ProviderUnavailable):
        p.refresh(None)
