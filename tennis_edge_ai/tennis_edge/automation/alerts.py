"""Alerts for significant projection changes. Delivered to the DB and optionally a webhook you own."""
from __future__ import annotations

import json
import logging

import httpx

from tennis_edge.config import get_settings
from tennis_edge.db.database import Database, utcnow_iso

log = logging.getLogger(__name__)


def raise_alert(db: Database, kind: str, message: str, payload: dict | None = None,
                client: httpx.Client | None = None) -> int:
    aid = db.execute("INSERT INTO alerts(ts, kind, message, payload) VALUES (?,?,?,?)",
                     (utcnow_iso(), kind, message, json.dumps(payload or {}, default=str)))
    url = get_settings().alert_webhook_url
    if url:
        try:
            c = client or httpx.Client(timeout=10)
            try:
                r = c.post(url, json={"text": f"[TENNIS EDGE AI] {message}", "kind": kind, "payload": payload})
                r.raise_for_status()
            finally:
                if client is None:
                    c.close()
            db.execute("UPDATE alerts SET delivered=1 WHERE id=?", (aid,))
        except Exception as exc:  # alert delivery must never break a job
            log.warning("alert webhook failed: %s", exc)
    return int(aid)


def line_move_alerts(db: Database, moves: list[dict]) -> int:
    for m in moves:
        raise_alert(db, "line_move", f"{m['player']} {m['market']}: line {m['from']} -> {m['to']} ({m['delta']:+.1f})", m)
    return len(moves)
