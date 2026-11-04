import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("TE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("TE_BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("TE_IMPORT_DIR", str(tmp_path / "imports"))
    from tennis_edge.config import reset_settings_cache
    reset_settings_cache()
    from tennis_edge.db.database import Database
    db = Database(tmp_path / "test.db")
    yield db
    reset_settings_cache()
