"""Automated jobs. None of these place wagers or touch betting accounts."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime

from tennis_edge.automation.alerts import line_move_alerts
from tennis_edge.betting.evaluator import evaluate_current
from tennis_edge.config import get_settings
from tennis_edge.db.database import Database, utcnow, utcnow_iso
from tennis_edge.providers.projections_csv import CSVProjectionProvider
from tennis_edge.providers.registry import refreshable_providers
from tennis_edge.tracking.tracker import TrackingError, auto_grade, capture_closing_lines, save_recommendation

log = logging.getLogger(__name__)


def refresh_sources(db: Database) -> dict:
    """Pull from every CONFIGURED automatic provider. Unconfigured providers are skipped (UNAVAILABLE)."""
    out = {}
    for p in refreshable_providers():
        started = utcnow_iso()
        try:
            res = p.refresh(db)
            out[p.name] = {k: v for k, v in res.items() if k != "line_moves"}
            line_move_alerts(db, res.get("line_moves", []))
        except Exception as exc:
            db.log_source(p.name, "refresh", "error", 0, str(exc)[:500], started)
            out[p.name] = {"error": str(exc)}
    return out


def import_dropbox(db: Database) -> dict:
    """Import CSVs dropped in var/imports/. File name must start with 'prizepicks' or 'tailgate'."""
    folder = get_settings().import_dropbox_dir
    done = folder / "processed"
    done.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in sorted(folder.glob("*.csv")):
        name = f.name.lower()
        platform = "PrizePicks" if name.startswith("prizepicks") else "Tailgate" if name.startswith("tailgate") else None
        if platform is None:
            out[f.name] = "skipped: name must start with 'prizepicks' or 'tailgate'"
            continue
        try:
            res = CSVProjectionProvider(platform).import_file(db, f)
            line_move_alerts(db, res.get("line_moves", []))
            out[f.name] = {"valid": res["valid_rows"], "errors": len(res["errors"])}
            shutil.move(str(f), done / f"{utcnow().strftime('%Y%m%dT%H%M%S')}_{f.name}")
        except Exception as exc:
            out[f.name] = f"error: {exc}"
    return out


def generate_recommendations(db: Database, now: datetime | None = None, stake: float = 0.0) -> dict:
    """Evaluate all upcoming projections, log model predictions, store BET-rated ones before start."""
    now = now or utcnow()
    opps = evaluate_current(db, upcoming_only=True, now=now)
    saved, skipped = 0, 0
    for o in opps:
        db.execute("INSERT INTO model_predictions(created_at,model_version,tour,market,player_id,opponent_id,start_time,"
                   "line,prob_over,projection_mean,details_json,projection_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (utcnow_iso(), o.model_version, o.tour, o.market, o.player_id, o.opponent_id, o.start_time, o.line,
                    o.prob_over if o.prob_over is not None else o.model_prob, o.model_mean,
                    '{"confidence": "%s", "no_bet": %d}' % (o.confidence, len(o.no_bet_reasons)), o.projection_key))
        if o.is_bet:
            try:
                save_recommendation(db, o, stake=stake, now=now)
                saved += 1
            except TrackingError:
                skipped += 1
    return {"evaluated": len(opps), "bets": sum(o.is_bet for o in opps), "saved": saved, "skipped": skipped}


def grade_and_close(db: Database) -> dict:
    closing = capture_closing_lines(db)
    graded = auto_grade(db)
    return {"closing_lines": closing, "graded": graded["graded"], "waiting": graded["waiting"]}


def run_all(db: Database) -> dict:
    return {"refresh": refresh_sources(db), "dropbox": import_dropbox(db),
            "recommendations": generate_recommendations(db), "grading": grade_and_close(db)}
