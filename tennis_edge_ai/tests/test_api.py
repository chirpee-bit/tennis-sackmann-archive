from fastapi.testclient import TestClient


def test_health_and_auth(tmp_db, monkeypatch):
    monkeypatch.setenv("TE_API_TOKEN", "secret-token-123")
    monkeypatch.setenv("ODDS_API_KEY", "abcdefghijk")
    from tennis_edge.config import reset_settings_cache
    reset_settings_cache()
    import tennis_edge.db.database as dbm
    dbm._default_db = None
    from tennis_edge.api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/providers").status_code == 401
    r = c.get("/providers", headers={"X-API-Token": "secret-token-123"})
    assert r.status_code == 200
    body = r.text
    assert "abcdefghijk" not in body and "secret-token-123" not in body   # credentials never exposed
    assert c.get("/performance/summary", headers={"X-API-Token": "secret-token-123"}).json()["total_recommendations"] == 0
    r = c.post("/predict/match", headers={"X-API-Token": "secret-token-123"},
               json={"tour": "ATP", "player_a_id": "ATP:1", "player_b_id": "ATP:2", "start_time": "2030-01-01T00:00:00Z"})
    assert r.status_code == 503      # no models in the temp environment -> explicit unavailability
    dbm._default_db = None


def test_public_view_masks_secrets(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "supersecretkey99")
    from tennis_edge.config import Settings
    view = Settings().public_view()
    assert "supersecretkey99" not in str(view)
    assert view["The Odds API key"].startswith("configured")
