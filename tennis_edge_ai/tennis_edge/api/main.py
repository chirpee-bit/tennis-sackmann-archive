"""FastAPI backend.

    uvicorn tennis_edge.api.main:app --port 8000

If ``TE_API_TOKEN`` is set, every request (except /health) must send the header
``X-API-Token``. Secrets are never returned by any endpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from tennis_edge import __version__
from tennis_edge.analytics.performance import tracked_summary
from tennis_edge.betting.evaluator import evaluate_current
from tennis_edge.config import get_settings
from tennis_edge.db.database import get_db
from tennis_edge.models.predict import get_predictor
from tennis_edge.providers.registry import all_providers
from tennis_edge.tracking.tracker import TrackingError, grade

app = FastAPI(title="TENNIS EDGE AI", version=__version__,
              description="Tennis statistical predictions and tracking. Never places wagers.")


def auth(x_api_token: str | None = Header(default=None)) -> None:
    token = get_settings().api_token
    if token and x_api_token != token:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Token")


@app.get("/health")
def health() -> dict:
    db = get_db()
    return {"status": "ok", "version": __version__, "matches": db.scalar("SELECT COUNT(*) FROM matches"),
            "active_models": [r["version"] for r in db.query("SELECT version FROM model_versions WHERE is_active=1")]}


@app.get("/providers", dependencies=[Depends(auth)])
def providers() -> list[dict]:
    return [p.status().as_dict() for p in all_providers()]


class MatchRequest(BaseModel):
    tour: str = Field(pattern="^(ATP|WTA)$")
    player_a_id: str
    player_b_id: str
    start_time: datetime
    surface: str = Field(default="Hard", pattern="^(Hard|Clay|Grass|Carpet)$")
    best_of: int = Field(default=3, ge=3, le=5)


@app.post("/predict/match", dependencies=[Depends(auth)])
def predict_match(req: MatchRequest) -> dict:
    pred = get_predictor(get_db(), req.tour)
    if not pred.ready:
        raise HTTPException(503, pred.why_not_ready())
    st = req.start_time if req.start_time.tzinfo else req.start_time.replace(tzinfo=timezone.utc)
    try:
        mp = pred.predict(req.player_a_id, req.player_b_id, st, req.surface, req.best_of)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"model_version": mp.model_version, "p_a_wins": mp.p_a_wins,
            "expected_total_games": float(mp.markov["total_games_mean"][0]),
            "expected_sets": float(mp.markov["sets_mean"][0]),
            "expected_games": {"a": float(mp.markov["a_games_mean"][0]), "b": float(mp.markov["b_games_mean"][0])},
            "expected_aces": {s: float(mp.props["aces"][s][0]) for s in ("a", "b")},
            "expected_double_faults": {s: float(mp.props["double_faults"][s][0]) for s in ("a", "b")},
            "expected_breaks": {s: float(mp.props["break_points_won"][s][0]) for s in ("a", "b")},
            "data_quality": mp.quality}


@app.get("/opportunities", dependencies=[Depends(auth)])
def opportunities(platform: str | None = None, market: str | None = None, bets_only: bool = False) -> list[dict]:
    opps = evaluate_current(get_db(), [platform] if platform else None, [market] if market else None)
    return [o.to_dict() for o in opps if (o.is_bet or not bets_only)]


@app.get("/recommendations", dependencies=[Depends(auth)])
def recommendations(status: str | None = None, limit: int = 200) -> list[dict]:
    sql = "SELECT * FROM bet_recommendations" + (" WHERE status=?" if status else "") + " ORDER BY created_at DESC LIMIT ?"
    return get_db().query(sql, ((status, limit) if status else (limit,)))


class GradeRequest(BaseModel):
    outcome: str | None = Field(default=None, pattern="^(win|loss|push|void)$")
    result_value: float | None = None
    notes: str | None = None


@app.post("/recommendations/{rec_id}/grade", dependencies=[Depends(auth)])
def grade_rec(rec_id: int, req: GradeRequest) -> dict:
    try:
        return grade(get_db(), rec_id, req.outcome, req.result_value, source="manual:api", notes=req.notes, actor="api")
    except TrackingError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/performance/summary", dependencies=[Depends(auth)])
def performance_summary() -> dict:
    return tracked_summary(get_db())
