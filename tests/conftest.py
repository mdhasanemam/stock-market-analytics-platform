import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def make_prices(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Synthetic business-day OHLCV random walk (deterministic)."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.015, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_provider="csv", database_url="sqlite:///:memory:", cache_dir=tmp_path / "cache", rate_limit_per_minute=100000)


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))
