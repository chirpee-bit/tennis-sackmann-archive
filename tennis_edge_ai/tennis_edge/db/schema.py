"""SQLite schema for TENNIS EDGE AI.

Design notes
------------
* Every natural record has a UNIQUE constraint so repeated imports are idempotent
  (duplicate prevention).
* Recommendations are immutable after creation; grading writes to
  ``settled_results`` and the ``audit_log`` records every state change.
* Player ids are namespaced by tour (``ATP:104925``) because Sackmann ATP and WTA
  id spaces are independent.
"""

SCHEMA_VERSION = 1

DDL = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id   TEXT PRIMARY KEY,          -- e.g. ATP:104925
    tour        TEXT NOT NULL CHECK (tour IN ('ATP','WTA')),
    source_id   TEXT NOT NULL,
    first_name  TEXT,
    last_name   TEXT,
    full_name   TEXT NOT NULL,
    name_norm   TEXT NOT NULL,
    hand        TEXT,
    dob         TEXT,
    ioc         TEXT,
    height_cm   REAL,
    UNIQUE (tour, source_id)
);
CREATE INDEX IF NOT EXISTS ix_players_name ON players(name_norm);

CREATE TABLE IF NOT EXISTS tournaments (
    tourney_key TEXT PRIMARY KEY,          -- tour:tourney_id
    tour        TEXT NOT NULL,
    tourney_id  TEXT NOT NULL,
    name        TEXT,
    surface     TEXT,
    draw_size   INTEGER,
    level       TEXT,
    start_date  TEXT,                      -- ISO date (tournament week start)
    UNIQUE (tour, tourney_id)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    tour            TEXT NOT NULL,
    tourney_key     TEXT NOT NULL REFERENCES tournaments(tourney_key),
    tourney_date    TEXT NOT NULL,
    est_match_date  TEXT NOT NULL,          -- estimated from round (Sackmann has no per-match date)
    round           TEXT,
    round_order     INTEGER,
    match_num       INTEGER,
    best_of         INTEGER,
    surface         TEXT,
    level           TEXT,
    winner_id       TEXT NOT NULL,
    loser_id        TEXT NOT NULL,
    score           TEXT,
    status          TEXT NOT NULL,          -- completed / retired / walkover / default / unknown
    total_games     INTEGER,
    total_sets      INTEGER,
    first_set_games INTEGER,
    winner_games    INTEGER,
    loser_games     INTEGER,
    tiebreaks       INTEGER,
    minutes         REAL,
    winner_rank     REAL,
    loser_rank      REAL,
    winner_rank_pts REAL,
    loser_rank_pts  REAL,
    winner_age      REAL,
    loser_age       REAL,
    source_file     TEXT,
    content_hash    TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_matches_date ON matches(tour, est_match_date);
CREATE INDEX IF NOT EXISTS ix_matches_winner ON matches(winner_id);
CREATE INDEX IF NOT EXISTS ix_matches_loser ON matches(loser_id);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id    TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    player_id   TEXT NOT NULL,
    is_winner   INTEGER NOT NULL,
    aces        REAL, double_faults REAL, serve_points REAL, first_in REAL,
    first_won   REAL, second_won REAL, service_games REAL,
    bp_saved    REAL, bp_faced REAL,
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS rankings (
    tour         TEXT NOT NULL,
    ranking_date TEXT NOT NULL,
    player_id    TEXT NOT NULL,
    rank         INTEGER,
    points       REAL,
    PRIMARY KEY (tour, ranking_date, player_id)
);

CREATE TABLE IF NOT EXISTS scheduled_matches (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tour           TEXT NOT NULL,
    player_a_name  TEXT NOT NULL,
    player_b_name  TEXT NOT NULL,
    player_a_id    TEXT,
    player_b_id    TEXT,
    start_time     TEXT NOT NULL,           -- ISO UTC
    tournament     TEXT,
    surface        TEXT,
    level          TEXT,
    best_of        INTEGER DEFAULT 3,
    source         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    UNIQUE (tour, player_a_name, player_b_name, start_time)
);

CREATE TABLE IF NOT EXISTS projections_current (
    projection_key TEXT PRIMARY KEY,        -- platform|market|player|opponent|start
    platform       TEXT NOT NULL,
    market         TEXT NOT NULL,
    tour           TEXT,
    player_name    TEXT NOT NULL,
    player_id      TEXT,
    opponent_name  TEXT,
    opponent_id    TEXT,
    line           REAL,
    over_odds      REAL,                     -- American odds when the platform provides them
    under_odds     REAL,
    payout_structure TEXT,
    start_time     TEXT,
    tournament     TEXT,
    surface        TEXT,
    best_of        INTEGER,
    source         TEXT NOT NULL,
    fetched_at     TEXT NOT NULL,
    first_seen_at  TEXT NOT NULL,
    first_line     REAL,
    raw_json       TEXT
);

CREATE TABLE IF NOT EXISTS projection_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_key TEXT NOT NULL,
    platform       TEXT NOT NULL,
    market         TEXT NOT NULL,
    player_name    TEXT NOT NULL,
    opponent_name  TEXT,
    line           REAL,
    over_odds      REAL,
    under_odds     REAL,
    start_time     TEXT,
    source         TEXT NOT NULL,
    fetched_at     TEXT NOT NULL,
    content_hash   TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_projhist_key ON projection_history(projection_key, fetched_at);

CREATE TABLE IF NOT EXISTS model_versions (
    version       TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    tour          TEXT NOT NULL,
    market        TEXT NOT NULL,
    model_type    TEXT NOT NULL,
    params_json   TEXT,
    train_start   TEXT,
    train_end     TEXT,
    metrics_json  TEXT,
    artifact_path TEXT,
    is_active     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS backtest_predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    tour          TEXT NOT NULL,
    market        TEXT NOT NULL,
    match_id      TEXT NOT NULL,
    match_date    TEXT NOT NULL,
    surface       TEXT,
    level         TEXT,
    best_of       INTEGER,
    line          REAL,
    prob          REAL,                      -- P(outcome==1), e.g. P(over) / P(player A wins)
    outcome       REAL,                      -- realised 0/1
    pred_mean     REAL,
    actual        REAL,
    UNIQUE (run_id, model_name, market, match_id, line)
);
CREATE INDEX IF NOT EXISTS ix_bt_run ON backtest_predictions(run_id, market, model_name);

CREATE TABLE IF NOT EXISTS model_predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    model_version  TEXT,
    tour           TEXT,
    market         TEXT NOT NULL,
    player_id      TEXT,
    opponent_id    TEXT,
    start_time     TEXT,
    line           REAL,
    prob_over      REAL,
    projection_mean REAL,
    details_json   TEXT,
    projection_key TEXT
);

CREATE TABLE IF NOT EXISTS bet_recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    platform        TEXT NOT NULL,
    market          TEXT NOT NULL,
    tour            TEXT,
    player_name     TEXT NOT NULL,
    player_id       TEXT,
    opponent_name   TEXT,
    opponent_id     TEXT,
    line            REAL,
    side            TEXT NOT NULL,           -- over / under / win
    odds_american   REAL,
    payout_structure TEXT,
    payout_snapshot TEXT,                     -- JSON copy of the payout table at creation time
    model_prob      REAL,
    implied_prob    REAL,
    breakeven_prob  REAL,
    fair_odds_american REAL,
    edge            REAL,
    ev_per_unit     REAL,
    uncertainty     REAL,
    confidence      TEXT,
    rank_score      REAL,
    stake           REAL NOT NULL DEFAULT 0,
    start_time      TEXT,
    model_version   TEXT,
    projection_key  TEXT,
    reasons_for     TEXT,
    reasons_against TEXT,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','win','loss','push','void')),
    result_value    REAL,
    graded_at       TEXT,
    grade_source    TEXT,
    closing_line    REAL,
    closing_odds    REAL,
    UNIQUE (platform, market, player_name, opponent_name, line, side, start_time)
);
CREATE INDEX IF NOT EXISTS ix_rec_status ON bet_recommendations(status);

CREATE TABLE IF NOT EXISTS settled_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER NOT NULL UNIQUE REFERENCES bet_recommendations(id),
    outcome           TEXT NOT NULL CHECK (outcome IN ('win','loss','push','void')),
    result_value      REAL,
    profit            REAL NOT NULL,
    settled_at        TEXT NOT NULL,
    source            TEXT NOT NULL,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS pickem_entries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    platform         TEXT NOT NULL,
    structure_id     TEXT NOT NULL,
    payout_snapshot  TEXT NOT NULL,
    stake            REAL NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    payout           REAL,
    profit           REAL,
    settled_at       TEXT
);

CREATE TABLE IF NOT EXISTS entry_legs (
    entry_id          INTEGER NOT NULL REFERENCES pickem_entries(id) ON DELETE CASCADE,
    recommendation_id INTEGER NOT NULL REFERENCES bet_recommendations(id),
    PRIMARY KEY (entry_id, recommendation_id)
);

CREATE TABLE IF NOT EXISTS bankroll_transactions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('deposit','withdrawal','stake','settlement','adjustment')),
    amount    REAL NOT NULL,
    reference TEXT,
    note      TEXT
);

CREATE TABLE IF NOT EXISTS data_source_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT NOT NULL,
    action      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    rows        INTEGER,
    message     TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    entity    TEXT,
    entity_id TEXT,
    details   TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    kind      TEXT NOT NULL,
    message   TEXT NOT NULL,
    payload   TEXT,
    delivered INTEGER NOT NULL DEFAULT 0
);

-- Recommendations may not be edited after creation except for grading fields.
CREATE TRIGGER IF NOT EXISTS trg_rec_immutable
BEFORE UPDATE OF platform, market, player_name, opponent_name, line, side, odds_american,
                 payout_snapshot, model_prob, created_at, stake, start_time
ON bet_recommendations
BEGIN
    SELECT RAISE(ABORT, 'bet_recommendations core fields are immutable');
END;
"""
