"""Application settings, loaded from environment variables / .env (never hard-code secrets)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Stock Market Analytics Platform"
    environment: str = "development"
    log_level: str = "INFO"

    # Market-data provider: "yfinance" (live, needs internet) or "csv" (bundled historical sample).
    data_provider: str = "yfinance"
    sample_data_dir: Path = ROOT / "data" / "sample"
    cache_dir: Path = ROOT / "data" / "cache"
    cache_ttl_minutes: int = 30

    database_url: str = f"sqlite:///{ROOT / 'data' / 'app.db'}"
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    rate_limit_per_minute: int = 120

    # ML defaults
    ml_train_window: int = 1500      # max rows used to fit a model
    ml_refit_every: int = 25         # walk-forward refit cadence (trading days)
    ml_max_test_days: int = 250      # max length of the out-of-sample backtest

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
