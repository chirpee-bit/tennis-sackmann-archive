"""Generic adapter for a LICENSED projection feed (PrizePicks or Tailgate).

PrizePicks publishes no public developer API. Use this adapter only if you hold
written permission / a data licence from the platform or an authorised reseller
that delivers JSON in this application's projection schema::

    [{"player": "...", "opponent": "...", "market": "Total Games", "line": 22.5,
      "start_time": "2026-11-05T14:00:00Z", "tour": "ATP", "surface": "Hard",
      "over_odds": null, "under_odds": null, "payout_structure": "PP_POWER_2"}, ...]

Configured with ``<PLATFORM>_API_URL`` and ``<PLATFORM>_API_KEY`` (sent as a
Bearer token). Without both, the provider reports UNAVAILABLE and nothing is fetched.
"""
from __future__ import annotations

import httpx
import pandas as pd

from tennis_edge.config import get_settings
from tennis_edge.db.database import Database
from tennis_edge.ingest.projections import store_projections
from tennis_edge.ingest.validation import validate_projection_frame
from tennis_edge.providers.base import DataProvider, ProviderStatus, ProviderUnavailable


class LicensedFeedProvider(DataProvider):
    kind = "projections"

    def __init__(self, platform: str, url: str | None = None, key: str | None = None, client: httpx.Client | None = None):
        s = get_settings()
        self.platform = platform
        self.name = f"{platform} licensed feed"
        if platform == "PrizePicks":
            self.url, self.key = url or s.prizepicks_api_url, key or s.prizepicks_api_key
            self.requires = ["PRIZEPICKS_LICENSED_API_URL + PRIZEPICKS_LICENSED_API_KEY (written data licence required)"]
        else:
            self.url, self.key = url or s.tailgate_api_url, key or s.tailgate_api_key
            self.requires = ["TAILGATE_API_URL + TAILGATE_API_KEY (authorised API access required)"]
        self.terms_note = "Only with explicit authorisation from the platform"
        self.client = client

    def status(self) -> ProviderStatus:
        if not (self.url and self.key):
            return self._status(False, "UNAVAILABLE - no licensed endpoint configured")
        return self._status(True, "Endpoint and key configured")

    def refresh(self, db: Database) -> dict:
        if not (self.url and self.key):
            raise ProviderUnavailable(f"{self.name} is not configured")
        client = self.client or httpx.Client(timeout=20)
        try:
            resp = client.get(self.url, headers={"Authorization": f"Bearer {self.key}"})
        finally:
            if self.client is None:
                client.close()
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("licensed feed must return a JSON list of projection objects")
        rows, errors = validate_projection_frame(pd.DataFrame(data).astype(str), self.platform)
        summary = store_projections(db, rows, source=self.name) if rows else {"inserted": 0, "updated": 0}
        summary["errors"] = errors
        db.log_source(self.name, "refresh", "ok", len(rows), f"errors={len(errors)}")
        return summary
