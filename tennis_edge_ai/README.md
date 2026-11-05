# TENNIS EDGE AI

Tennis-only statistical modelling, opportunity evaluation and result tracking for PrizePicks / Tailgate style
projections and sportsbook prices. It predicts match and player statistics, compares them with the lines you
import, ranks opportunities (or returns **NO BET**), stores every recommendation before the match, grades
results from verified data, and measures its own out-of-sample accuracy.

> **No guarantees.** The models are statistical estimates validated out-of-sample; they can and will lose.
> The app never places wagers, never logs into betting accounts and never scrapes any platform.

---

## 1. Installation

Requires Python 3.12+ (tested on 3.13).

```bash
cd tennis_edge_ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # optional: edit settings / add API keys
```

## 2. Build the data and models (one-time, ~1 hour on 4 cores)

Run from `tennis_edge_ai/`:

```bash
python scripts/ingest_history.py        # Phase 2: load ATP/WTA history into SQLite      (~5 min)
python scripts/build_features.py        # Phase 3: leakage-free pre-match features         (~5 min)
python scripts/train_models.py          # Phases 3-4: walk-forward validation + production (~30-60 min)
```

Train tours in parallel to halve the time:
`python scripts/train_models.py --tours ATP & python scripts/train_models.py --tours WTA`
(`--no-lgbm` skips the gradient-boosting challenger; `TE_LGBM_THREADS` sets its threads per process.)

## 3. Run

```bash
streamlit run app/Home.py                               # dashboard  -> http://localhost:8501 (run from tennis_edge_ai/)
uvicorn tennis_edge.api.main:app --port 8000            # REST API   -> http://localhost:8000/docs
python -m tennis_edge.automation.scheduler              # scheduled refresh / recommend / grade
python -m tennis_edge.automation.scheduler --once       # run every job once
python -m pytest                                         # test suite
```

Optional demo data, clearly labelled **SIMULATED** (fake lines, real player names):
`python scripts/make_simulated_projections.py` writes a CSV into `var/imports/` that the scheduler or the
Data Management page can import. Never treat simulated lines as real.

---

## 4. Data sources and what you need

| Source | Status | What you need |
|---|---|---|
| Historical results, serve/return stats, rankings (Jeff Sackmann ATP/WTA) | **Included** (`../atp`, `../wta`) | Nothing. Licence CC BY-NC-SA 4.0 → **non-commercial use only**, attribute Jeff Sackmann. |
| PrizePicks projections | CSV upload / manual entry | PrizePicks has **no public developer API**. Automated access needs a written data licence from PrizePicks or an authorised reseller → set `PRIZEPICKS_LICENSED_API_URL` + `PRIZEPICKS_LICENSED_API_KEY` (feed must return the JSON schema below). |
| Tailgate projections / odds | CSV upload / manual entry | Authorised API access only → `TAILGATE_API_URL` + `TAILGATE_API_KEY`. |
| Sportsbook odds (optional) | Off until configured | `ODDS_API_KEY` from <https://the-odds-api.com> (free tier ≈ 500 requests / month; paid plans for more). Tennis keys are per-tournament and only active during events. |
| Injuries / withdrawals, schedules, conditions | **UNAVAILABLE** | No reliable licensed source is configured. The app does not guess them; enter scheduled matches manually (Match Analysis page) or via projection CSVs. |
| Alerts (optional) | Off until configured | `TE_ALERT_WEBHOOK_URL` — a Slack/Discord/other webhook *you* own. |

Unconfigured providers report **UNAVAILABLE** on the Data Management page; nothing is fabricated.

### Updating historical data

The bundled archive ends **late May 2026**. Opportunities for players whose last recorded match is older than
`TE_MAX_DATA_AGE_DAYS` (default 120) are returned as **NO BET**, so with stale data most current projections will be
NO BET — that is intended. To update:

```bash
git clone https://github.com/JeffSackmann/tennis_atp ../atp_new   # or pull into the existing folders
# copy/replace the CSVs into ../atp and ../wta (or point TE_SACKMANN_DIR at a folder holding atp/ and wta/)
python scripts/ingest_history.py && python scripts/build_features.py && python scripts/train_models.py
```

---

## 5. Projection CSV format

Header names are matched case-insensitively with common aliases (`player_name`, `stat_type`, `line_score`, ...).

| column | required | notes |
|---|---|---|
| `player` | yes | full name; matched to the historical player table (accents/order tolerant; ambiguous → rejected) |
| `opponent` | yes for pricing | required for every market (the model prices a matchup) |
| `market` | yes | `Total Games`, `Games Won` (player), `1st Set Games`, `Total Sets`, `Aces`, `Double Faults`, `Break Points Won`, `Match Winner` |
| `line` | yes (except match winner) | plausibility-checked per market |
| `start_time` | yes | ISO 8601, UTC if no zone |
| `tour` | recommended | `ATP` or `WTA` (other tours rejected: insufficient data) |
| `surface`, `best_of` | recommended | defaults Hard / 3 are **flagged** in the reasons |
| `over_odds`, `under_odds` | optional | American odds. For `Match Winner`: price on `player` / price on `opponent` |
| `payout_structure` | optional | id from `config/payouts.json` (pick'em) |
| `platform` | if not chosen in the UI | `PrizePicks` / `Tailgate` |

Validation: `.csv` only, ≤10 MB, ≤20 000 rows, UTF-8, formula-injection neutralised, per-row Pydantic validation,
duplicate rows rejected, imports idempotent (history keeps every distinct snapshot for line-movement / CLV).

## 6. Payout structures and markets — you must verify them

* `config/payouts.json` ships **example, unverified** pick'em payout tables. Expected value and break-even
  probabilities are computed **only** for structures you mark `verified: true` (Data Management → Payouts) after
  checking the multipliers in your own app — they differ by platform, state and date.
* `config/markets.json` records which markets *you* have verified each platform offers. Nothing is assumed.

---

## 7. How it works

**Features** (`tennis_edge/features/engine.py`): one chronological pass over ~835k matches. For every match the
engine reads both players' state *before* the match, emits features, then updates the state — so features can
only use earlier matches (unit-tested). The same engine snapshot produces live features.
Surface-blended Elo (538 K-schedule), time-decayed (300-day half-life) serve/return rates shrunk to tour/surface
averages, opponent-adjusted serve/return residuals, Elo-residual form (150-day half-life), surface win rates,
ace/DF/break-point rates, H2H, rest days and 7/14-day workload, ranking, age, height, best-of-3/5.
Sackmann files have no per-match dates: match dates are **estimated** from the round.

**Models**
* Match winner — symmetric logistic regression on A−B differences (mirror augmentation, no intercept) with
  temperature calibration; LightGBM is evaluated as a challenger and only used if it beats logistic in walk-forward
  log loss with z < −2; Elo and ranking baselines for comparison. Separate ATP and WTA models.
* Total games, player games won, first-set games, total sets — exact Markov chain of the match (point → game →
  tiebreak → set → match, best-of-3/5, serve order across sets) with Barnett–Clarke serve probabilities. The
  serve-dominance term is solved so the chain reproduces the calibrated win probability, and **random effects**
  (Gauss–Hermite quadrature over dominance and hold level, fitted by likelihood on training data) remove the
  systematic over-prediction of match length that a plug-in Markov model has (Jensen's inequality).
* Aces, double faults, break points won — Poisson GLMs with Markov-derived exposure (expected service points /
  opponent service games) and negative-binomial dispersion chosen by likelihood.
* Isotonic line-calibration per market, kept only if it improves pooled out-of-sample Brier in a nested walk-forward.

**Validation** (`tennis_edge/validation/walk_forward.py`): for each test year Y, train on the previous 10 years,
calibrate on out-of-sample predictions for Y−1, test on Y. No random splits. Duplicate matches removed at
ingestion; outcome order randomised by hash; player ids never used as features. Metrics: accuracy, Brier, log loss,
ECE, MAE, RMSE, by tour / market / surface / level / format / quarter, paired Diebold–Mariano tests vs baselines,
drift (recent-window Brier z-test, feature PSI). Count-market backtests use **synthetic lines** (no historical
platform lines exist in the data) — real lines are sharper, so live edges will be smaller.

**Evaluator / ranking** (`tennis_edge/betting/evaluator.py`): model probability per side; two-way margin removed
(power method; multiplicative and Shin available); pick'em break-even from verified payout tables; EV only when the
payout is actually known; fair odds; projection difference; heuristic uncertainty
`sqrt(ECE² + p(1−p)/(n+10))`; data freshness; sample size. A selection is a bet only if the **lower bound** of its edge
clears `TE_MIN_EDGE` and data/sample/market-validation checks pass. Score = edge lower bound × reliability
(freshness, sample) × validated market skill (backtest Brier skill, live ROI once ≥30 settled) × line-movement
factor. Otherwise **NO BET** with the reasons. Same-match selections are flagged as correlated.

**Tracking** (`tennis_edge/tracking/tracker.py`): recommendations can only be saved before the start time; core
fields are immutable (SQLite trigger) and every action is in `audit_log`. Win / loss / push / void are separate;
P&L uses the stored odds or the stored payout-table snapshot; voided pick'em legs re-price the entry with the same
family's smaller structure (or refund). Auto-grading uses only results present in the verified historical database;
walkovers/retirements are void by default (`TE_RETIREMENT_RULE`). Closing lines are captured from the last snapshot
before start (CLV).

## 8. Project layout

```
tennis_edge_ai/
  app/                 Streamlit dashboard (Home = Overview + 5 pages)
  config/              markets.json, payouts.json (user-verified)
  scripts/             ingest_history, build_features, train_models, make_simulated_projections
  tennis_edge/
    providers/         Sackmann, CSV/manual, The Odds API, licensed feed, registry
    ingest/            score parser, historical loader, upload validation, projection store
    features/          chronological feature engine + builder
    models/            winner, markov, totals (random effects), props, training, predict
    validation/        metrics, walk-forward, drift
    betting/           odds/devig/EV, pick'em payouts, evaluator + ranking
    tracking/          recommendations, grading, entries, bankroll, CLV
    analytics/         performance summaries
    automation/        jobs, scheduler, alerts
    api/               FastAPI app
    db/                schema + database helpers (backup/export/audit)
  tests/               pytest suite
  var/                 database, artifacts, backups, imports (git-ignored)
```

## 9. Security

API keys live only in `.env` (git-ignored) and are masked everywhere in the UI/API. No sportsbook or platform
passwords are stored or requested. Uploads are validated; duplicate records are prevented by UNIQUE constraints;
the FastAPI app requires `X-API-Token` when `TE_API_TOKEN` is set.

## 10. Licence

Application code: MIT. Bundled historical data © Jeff Sackmann, CC BY-NC-SA 4.0 (non-commercial, share-alike).
