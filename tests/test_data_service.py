from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backend.app.core.errors import ProviderUnavailable
from backend.app.services.data_service import DataService, normalize_symbol, validate_ohlcv
from backend.app.services.providers import CsvProvider, DataProvider


def test_validate_drops_bad_rows_and_duplicates(prices):
    df = prices.iloc[:30].copy()
    df.iloc[3, df.columns.get_loc("Close")] = np.nan
    df.iloc[5, df.columns.get_loc("Open")] = -1
    df = pd.concat([df, df.iloc[[10]]])               # duplicate date
    out, warns = validate_ohlcv(df)
    assert len(out) == 28 and out.index.is_unique and out.index.is_monotonic_increasing
    assert any("Dropped 2" in w for w in warns)


def test_weekends_are_not_treated_as_missing(prices):
    out, warns = validate_ohlcv(prices.iloc[:100])
    assert not any("gap" in w for w in warns)
    assert len(out) == 100                            # nothing is interpolated or invented


def test_symbol_validation():
    assert normalize_symbol(" aapl ") == "AAPL"
    for bad in ["", "A B", "DROP;TABLE", "../etc", "X" * 30]:
        with pytest.raises(ValueError):
            normalize_symbol(bad)


class FlakyProvider(DataProvider):
    name = "flaky"

    def __init__(self, df):
        self.df, self.fail = df, False

    def history(self, symbol):
        if self.fail:
            raise ProviderUnavailable("rate limited")
        return self.df


def test_stale_cache_is_served_when_provider_fails(prices, settings):
    prov = FlakyProvider(prices)
    ds = DataService(prov, settings)
    first = ds.get("TEST")
    assert not first.stale
    first.fetched_at = datetime.now(timezone.utc) - timedelta(hours=5)   # expire the cache
    prov.fail = True
    again = ds.get("TEST")
    assert again.stale and any("cached" in w for w in again.warnings)
    fresh = DataService(prov, settings.model_copy(update={"cache_dir": settings.cache_dir / "other"}))
    with pytest.raises(ProviderUnavailable):                              # nothing cached -> error surfaces
        fresh.get("TEST")


def test_slice_ranges_anchor_on_latest_bar(settings):
    ds = DataService(CsvProvider(settings.sample_data_dir), settings)
    d = ds.get("AAPL")
    month = ds.slice(d, "1M")
    assert month.index[-1] == d.df.index[-1] and len(month) < 30
    assert len(ds.slice(d, "1D")) == 2
    assert len(ds.slice(d, None, "2017-01-01", "2017-01-31")) > 15
    with pytest.raises(ValueError):
        ds.slice(d, "7Y")
    with pytest.raises(ValueError):
        ds.slice(d, None, "2017-02-01", "2017-01-01")
