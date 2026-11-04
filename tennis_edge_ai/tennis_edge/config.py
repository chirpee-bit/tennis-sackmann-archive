"""Application configuration.

All secrets and environment-specific settings come from environment variables
(optionally loaded from a ``.env`` file).  Nothing secret is ever hard-coded and
the :func:`Settings.public_view` helper is the only representation that may be
shown in the dashboard (credentials are masked).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _mask(secret: str | None) -> str:
    if not secret:
        return "not configured"
    return "configured (****" + secret[-2:] + ")" if len(secret) > 6 else "configured (****)"


@dataclass(frozen=True)
class Settings:
    # Storage
    db_path: Path = field(default_factory=lambda: Path(_env("TE_DB_PATH", str(PROJECT_ROOT / "var" / "tennis_edge.db"))))
    artifacts_dir: Path = field(default_factory=lambda: Path(_env("TE_ARTIFACTS_DIR", str(PROJECT_ROOT / "var" / "artifacts"))))
    backup_dir: Path = field(default_factory=lambda: Path(_env("TE_BACKUP_DIR", str(PROJECT_ROOT / "var" / "backups"))))
    import_dropbox_dir: Path = field(default_factory=lambda: Path(_env("TE_IMPORT_DIR", str(PROJECT_ROOT / "var" / "imports"))))
    payouts_file: Path = field(default_factory=lambda: Path(_env("TE_PAYOUTS_FILE", str(PROJECT_ROOT / "config" / "payouts.json"))))
    markets_file: Path = field(default_factory=lambda: Path(_env("TE_MARKETS_FILE", str(PROJECT_ROOT / "config" / "markets.json"))))

    # Historical data (Jeff Sackmann archive, CC BY-NC-SA 4.0)
    sackmann_dir: Path = field(default_factory=lambda: Path(_env("TE_SACKMANN_DIR", str(REPO_ROOT))))
    history_start_year: int = field(default_factory=lambda: _env_int("TE_HISTORY_START_YEAR", 2000))
    include_challengers: bool = field(default_factory=lambda: _env_bool("TE_INCLUDE_CHALLENGERS", True))
    include_itf: bool = field(default_factory=lambda: _env_bool("TE_INCLUDE_ITF", True))
    include_futures: bool = field(default_factory=lambda: _env_bool("TE_INCLUDE_FUTURES", False))

    # Model/validation
    first_test_year: int = field(default_factory=lambda: _env_int("TE_FIRST_TEST_YEAR", 2019))
    train_window_years: int = field(default_factory=lambda: _env_int("TE_TRAIN_WINDOW_YEARS", 10))
    random_seed: int = field(default_factory=lambda: _env_int("TE_RANDOM_SEED", 7))

    # Betting evaluator
    min_edge: float = field(default_factory=lambda: _env_float("TE_MIN_EDGE", 0.02))
    max_data_age_days: int = field(default_factory=lambda: _env_int("TE_MAX_DATA_AGE_DAYS", 120))
    min_sample_matches: int = field(default_factory=lambda: _env_int("TE_MIN_SAMPLE_MATCHES", 8))
    kelly_fraction: float = field(default_factory=lambda: _env_float("TE_KELLY_FRACTION", 0.25))

    # Automation
    refresh_minutes: int = field(default_factory=lambda: _env_int("TE_REFRESH_MINUTES", 30))
    grade_minutes: int = field(default_factory=lambda: _env_int("TE_GRADE_MINUTES", 60))
    alert_line_move: float = field(default_factory=lambda: _env_float("TE_ALERT_LINE_MOVE", 1.0))
    alert_webhook_url: str | None = field(default_factory=lambda: _env("TE_ALERT_WEBHOOK_URL"))

    # Credentials (never displayed in clear text)
    odds_api_key: str | None = field(default_factory=lambda: _env("ODDS_API_KEY"))
    odds_api_regions: str = field(default_factory=lambda: _env("ODDS_API_REGIONS", "us") or "us")
    prizepicks_api_key: str | None = field(default_factory=lambda: _env("PRIZEPICKS_LICENSED_API_KEY"))
    prizepicks_api_url: str | None = field(default_factory=lambda: _env("PRIZEPICKS_LICENSED_API_URL"))
    tailgate_api_key: str | None = field(default_factory=lambda: _env("TAILGATE_API_KEY"))
    tailgate_api_url: str | None = field(default_factory=lambda: _env("TAILGATE_API_URL"))
    api_token: str | None = field(default_factory=lambda: _env("TE_API_TOKEN"))

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.artifacts_dir, self.backup_dir, self.import_dropbox_dir):
            d.mkdir(parents=True, exist_ok=True)

    def public_view(self) -> dict[str, str]:
        """Settings safe to render in the UI (secrets masked)."""
        return {
            "Database": str(self.db_path),
            "Artifacts": str(self.artifacts_dir),
            "Historical data dir": str(self.sackmann_dir),
            "History start year": str(self.history_start_year),
            "Include Challengers / ITF / Futures": f"{self.include_challengers} / {self.include_itf} / {self.include_futures}",
            "First walk-forward test year": str(self.first_test_year),
            "Minimum edge for a bet": f"{self.min_edge:.1%}",
            "Max data age (days)": str(self.max_data_age_days),
            "The Odds API key": _mask(self.odds_api_key),
            "PrizePicks licensed feed": _mask(self.prizepicks_api_key),
            "Tailgate API": _mask(self.tailgate_api_key),
            "Alert webhook": "configured" if self.alert_webhook_url else "not configured",
            "API token": _mask(self.api_token),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()
