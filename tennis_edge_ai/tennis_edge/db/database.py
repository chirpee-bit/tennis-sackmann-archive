"""SQLite connection management, audit logging, backup and export."""
from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from tennis_edge.config import get_settings
from tennis_edge.db.schema import DDL, SCHEMA_VERSION


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().replace(microsecond=0).isoformat()


class Database:
    """Thin wrapper over sqlite3 with helpers used across the application."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else get_settings().db_path
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        if str(self.path) == ":memory:":
            self._memory_conn = self._new_conn()
        self.init_schema()

    # ------------------------------------------------------------------ connections
    def _new_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL") if str(self.path) != ":memory:" else None
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            try:
                yield self._memory_conn
                self._memory_conn.commit()
            except Exception:
                self._memory_conn.rollback()
                raise
            return
        conn = self._new_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(DDL)
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    # ------------------------------------------------------------------ helpers
    def execute(self, sql: str, params: tuple | dict = ()) -> int:
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return cur.lastrowid if cur.lastrowid else cur.rowcount

    def executemany(self, sql: str, rows: list[tuple] | list[dict]) -> int:
        if not rows:
            return 0
        with self.connect() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            return conn.total_changes - before

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: tuple | dict = ()) -> Any:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return None if row is None else row[0]

    def df(self, sql: str, params: tuple | dict = ()) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        v = self.scalar("SELECT value FROM meta WHERE key=?", (key,))
        return default if v is None else v

    def set_meta(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ audit / logs
    def audit(self, action: str, entity: str | None = None, entity_id: Any = None,
              details: dict | None = None, actor: str = "system") -> None:
        self.execute(
            "INSERT INTO audit_log(ts, actor, action, entity, entity_id, details) VALUES (?,?,?,?,?,?)",
            (utcnow_iso(), actor, action, entity, None if entity_id is None else str(entity_id),
             json.dumps(details, default=str) if details else None),
        )

    def log_source(self, provider: str, action: str, status: str, rows: int | None = None,
                   message: str | None = None, started_at: str | None = None) -> None:
        self.execute(
            "INSERT INTO data_source_logs(provider, action, started_at, finished_at, status, rows, message) "
            "VALUES (?,?,?,?,?,?,?)",
            (provider, action, started_at or utcnow_iso(), utcnow_iso(), status, rows, message),
        )

    # ------------------------------------------------------------------ backup / export
    def backup(self, dest_dir: str | Path | None = None) -> Path:
        dest_dir = Path(dest_dir) if dest_dir else get_settings().backup_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"tennis_edge_{utcnow().strftime('%Y%m%dT%H%M%SZ')}.db"
        with self.connect() as conn:
            target = sqlite3.connect(str(dest))
            try:
                conn.backup(target)
            finally:
                target.close()
        self.audit("backup", "database", None, {"path": str(dest)})
        return dest

    def export_table_csv(self, table: str, dest: str | Path) -> Path:
        allowed = {r["name"] for r in self.query("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in allowed:
            raise ValueError(f"Unknown table {table!r}")
        frame = self.df(f'SELECT * FROM "{table}"')
        dest = Path(dest)
        frame.to_csv(dest, index=False)
        self.audit("export", table, None, {"path": str(dest), "rows": len(frame)})
        return dest

    def table_counts(self) -> dict[str, int]:
        names = [r["name"] for r in self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {n: int(self.scalar(f'SELECT COUNT(*) FROM "{n}"')) for n in names}


_default_db: Database | None = None


def get_db() -> Database:
    global _default_db
    if _default_db is None or _default_db.path != get_settings().db_path:
        _default_db = Database()
    return _default_db


def copy_file(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)
