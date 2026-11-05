"""Central list of providers so the UI / scheduler can report and refresh them."""
from __future__ import annotations

from tennis_edge.providers.base import DataProvider
from tennis_edge.providers.licensed_feed import LicensedFeedProvider
from tennis_edge.providers.odds_api import OddsAPIProvider
from tennis_edge.providers.projections_csv import CSVProjectionProvider
from tennis_edge.providers.sackmann import SackmannProvider

PLATFORMS = ("PrizePicks", "Tailgate")


def all_providers() -> list[DataProvider]:
    return [SackmannProvider(), CSVProjectionProvider("PrizePicks"), CSVProjectionProvider("Tailgate"),
            LicensedFeedProvider("PrizePicks"), LicensedFeedProvider("Tailgate"), OddsAPIProvider()]


def refreshable_providers() -> list[DataProvider]:
    """Providers that can pull data automatically (only if configured)."""
    return [p for p in all_providers() if hasattr(p, "refresh") and p.status().available]
