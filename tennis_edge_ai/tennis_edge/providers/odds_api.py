"""The Odds API (https://the-odds-api.com) - licensed sportsbook odds.

Requires ``ODDS_API_KEY`` (free tier: limited monthly requests; paid tiers for
more). Tennis sport keys are tournament-specific (e.g. ``tennis_atp_us_open``)
and only active while that event runs; coverage of smaller events and of
totals markets varies. Only h2h (match winner) and totals (total games) are
requested. Each call costs quota; the remaining quota is logged.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from tennis_edge.config import get_settings
from tennis_edge.db.database import Database
from tennis_edge.ingest.projections import store_projections
from tennis_edge.providers.base import DataProvider, ProviderStatus, ProviderUnavailable

BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIProvider(DataProvider):
    name = "The Odds API"
    kind = "odds"
    requires = ["ODDS_API_KEY in .env (https://the-odds-api.com - free or paid plan)"]
    terms_note = "Licensed API - respect plan limits and terms of use"

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None, regions: str | None = None):
        s = get_settings()
        self.api_key = api_key if api_key is not None else s.odds_api_key
        self.regions = regions or s.odds_api_regions
        self.client = client
        self.remaining: str | None = None

    def status(self) -> ProviderStatus:
        if not self.api_key:
            return self._status(False, "UNAVAILABLE - ODDS_API_KEY not set")
        return self._status(True, "Key configured (connectivity checked on refresh)")

    def _get(self, path: str, **params) -> list | dict:
        if not self.api_key:
            raise ProviderUnavailable("ODDS_API_KEY not configured")
        client = self.client or httpx.Client(timeout=20)
        try:
            resp = client.get(f"{BASE_URL}{path}", params={"apiKey": self.api_key, **params})
        finally:
            if self.client is None:
                client.close()
        self.remaining = resp.headers.get("x-requests-remaining")
        if resp.status_code == 401:
            raise ProviderUnavailable("The Odds API rejected the key (401)")
        if resp.status_code == 429:
            raise ProviderUnavailable("The Odds API quota exhausted (429)")
        resp.raise_for_status()
        return resp.json()

    def tennis_sports(self) -> list[dict]:
        return [s for s in self._get("/sports") if str(s.get("key", "")).startswith("tennis_") and s.get("active")]

    @staticmethod
    def parse_events(events: list[dict], sport_key: str) -> list[dict]:
        tour = "WTA" if "_wta_" in sport_key else "ATP"
        out = []
        for ev in events:
            home, away = ev.get("home_team"), ev.get("away_team")
            start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(timezone.utc)
            for bk in ev.get("bookmakers", []):
                platform = f"Sportsbook:{bk.get('title', bk.get('key'))}"
                for mk in bk.get("markets", []):
                    outs = {o.get("name"): o for o in mk.get("outcomes", [])}
                    if mk.get("key") == "h2h" and home in outs and away in outs:
                        out.append({"platform": platform, "player": home, "opponent": away, "market": "match_winner",
                                    "line": None, "over_odds": float(outs[home]["price"]),
                                    "under_odds": float(outs[away]["price"]), "start_time": start,
                                    "tournament": ev.get("sport_title"), "tour": tour, "surface": None,
                                    "best_of": None, "payout_structure": None})
                    elif mk.get("key") == "totals" and "Over" in outs and "Under" in outs:
                        if outs["Over"].get("point") != outs["Under"].get("point"):
                            continue
                        out.append({"platform": platform, "player": home, "opponent": away, "market": "total_games",
                                    "line": float(outs["Over"]["point"]), "over_odds": float(outs["Over"]["price"]),
                                    "under_odds": float(outs["Under"]["price"]), "start_time": start,
                                    "tournament": ev.get("sport_title"), "tour": tour, "surface": None,
                                    "best_of": None, "payout_structure": None})
        return out

    def refresh(self, db: Database) -> dict:
        sports = self.tennis_sports()
        rows = []
        for sp in sports:
            events = self._get(f"/sports/{sp['key']}/odds", regions=self.regions, markets="h2h,totals",
                               oddsFormat="american", dateFormat="iso")
            rows.extend(self.parse_events(events, sp["key"]))
        summary = store_projections(db, rows, source=self.name) if rows else {"inserted": 0, "updated": 0}
        summary["sports"] = [s["key"] for s in sports]
        summary["requests_remaining"] = self.remaining
        db.log_source(self.name, "refresh", "ok", len(rows), f"sports={len(sports)} remaining={self.remaining}")
        return summary
