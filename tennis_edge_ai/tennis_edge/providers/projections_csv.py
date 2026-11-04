"""CSV / manual-entry adapters for PrizePicks and Tailgate projections.

These adapters read files that YOU create (by hand, or exported from a tool or
licensed provider you are entitled to use). They do not contact the platforms.
No scraping, no private endpoints.
"""
from __future__ import annotations

from pathlib import Path

from tennis_edge.db.database import Database
from tennis_edge.ingest.projections import store_projections
from tennis_edge.ingest.validation import read_upload, validate_projection_frame
from tennis_edge.providers.base import DataProvider, ProviderStatus


class CSVProjectionProvider(DataProvider):
    kind = "projections"
    requires = ["A CSV file you created (see README 'Projection CSV format')"]
    terms_note = "Manual / user-supplied data. Do not export data in violation of a platform's terms."

    def __init__(self, platform: str):
        self.platform = platform
        self.name = f"{platform} CSV import"

    def status(self) -> ProviderStatus:
        return self._status(True, "Ready - upload a CSV on the Data Management page or drop it in var/imports/")

    def import_bytes(self, db: Database, data: bytes, filename: str) -> dict:
        frame = read_upload(data, filename)
        rows, errors = validate_projection_frame(frame, self.platform)
        summary = store_projections(db, rows, source=f"{self.name}:{filename}") if rows else \
            {"inserted": 0, "updated": 0, "unchanged": 0, "line_moves": [], "unresolved": []}
        summary.update({"valid_rows": len(rows), "errors": errors})
        db.audit("upload_csv", "projections", filename, {"platform": self.platform, "valid": len(rows),
                                                          "errors": len(errors)}, actor="user")
        return summary

    def import_file(self, db: Database, path: Path) -> dict:
        return self.import_bytes(db, Path(path).read_bytes(), Path(path).name)


class ManualEntryProvider(CSVProjectionProvider):
    def __init__(self, platform: str):
        super().__init__(platform)
        self.name = f"{platform} manual entry"

    def status(self) -> ProviderStatus:
        return self._status(True, "Ready - enter projections on the Data Management page")

    def import_rows(self, db: Database, rows: list[dict]) -> dict:
        import pandas as pd
        valid, errors = validate_projection_frame(pd.DataFrame(rows).astype(str), self.platform)
        summary = store_projections(db, valid, source=self.name) if valid else \
            {"inserted": 0, "updated": 0, "unchanged": 0, "line_moves": [], "unresolved": []}
        summary.update({"valid_rows": len(valid), "errors": errors})
        return summary
