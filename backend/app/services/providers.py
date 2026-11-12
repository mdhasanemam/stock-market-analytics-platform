"""Pluggable market-data providers.

Every provider returns a DataFrame indexed by a tz-naive DatetimeIndex with the columns
Open, High, Low, Close, Volume. Swap providers via DATA_PROVIDER without touching callers.
To add one (Alpha Vantage, FMP, ...) subclass DataProvider and register it in get_provider().
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from backend.app.core.config import Settings
from backend.app.core.errors import ProviderUnavailable, SymbolNotFound

log = logging.getLogger(__name__)
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Small local directory used for autocomplete when a provider's search is unavailable.
POPULAR = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.", "META": "Meta Platforms, Inc.", "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation", "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson", "WMT": "Walmart Inc.", "XOM": "Exxon Mobil Corporation",
    "NFLX": "Netflix, Inc.", "KO": "The Coca-Cola Company", "DIS": "The Walt Disney Company",
}
CSV_NAMES = {**POPULAR, "FB": "Facebook, Inc. (now Meta Platforms)"}


class DataProvider(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def history(self, symbol: str) -> pd.DataFrame:
        """Full available daily history for `symbol`. Raise SymbolNotFound / ProviderUnavailable."""

    def search(self, query: str, limit: int = 8) -> list[dict]:
        q = query.strip().lower()
        out = [{"symbol": s, "name": n} for s, n in POPULAR.items() if q in s.lower() or q in n.lower()]
        return out[:limit]

    def company_name(self, symbol: str) -> str | None:
        return POPULAR.get(symbol)


class YFinanceProvider(DataProvider):
    name = "yfinance"
    description = (
        "Yahoo Finance via the community `yfinance` library. Prices are split/dividend adjusted. "
        "Data is unofficial, may be delayed, and this app uses daily bars. "
        "For research and education only."
    )

    def __init__(self, retries: int = 3, backoff: float = 1.5):
        self.retries, self.backoff = retries, backoff

    def history(self, symbol: str) -> pd.DataFrame:
        import yfinance as yf  # lazy import so offline mode/tests never touch the network

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                df = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
                if df is None or df.empty:
                    raise SymbolNotFound(f"No price data returned for '{symbol}'.")
                df = df[COLUMNS].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                return df
            except SymbolNotFound:
                raise
            except Exception as exc:  # network errors, rate limits, parsing problems
                last_exc = exc
                log.warning("yfinance attempt %d/%d failed for %s: %s", attempt, self.retries, symbol, exc)
                time.sleep(self.backoff * attempt)
        raise ProviderUnavailable(f"Yahoo Finance is unavailable or rate-limited ({last_exc}).")

    def search(self, query: str, limit: int = 8) -> list[dict]:
        try:
            import yfinance as yf

            quotes = yf.Search(query, max_results=limit).quotes
            res = [
                {"symbol": q["symbol"], "name": q.get("shortname") or q.get("longname") or q["symbol"]}
                for q in quotes
                if q.get("quoteType") in {"EQUITY", "ETF", "INDEX"} and "symbol" in q
            ]
            if res:
                return res[:limit]
        except Exception as exc:
            log.info("yfinance search failed (%s); using local directory", exc)
        return super().search(query, limit)


class CsvProvider(DataProvider):
    name = "csv"
    description = (
        "Bundled historical sample (AAPL, FB, MSFT, TSLA daily bars ending 2017-11-10) from the "
        "original stock-price-prediction project. Offline/demo mode: NOT current market data."
    )

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def _symbols(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.csv"))

    def history(self, symbol: str) -> pd.DataFrame:
        path = self.dir / f"{symbol}.csv"
        if not path.exists():
            raise SymbolNotFound(f"'{symbol}' is not in the bundled sample ({', '.join(self._symbols())}).")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df[COLUMNS]

    def search(self, query: str, limit: int = 8) -> list[dict]:
        q = query.strip().lower()
        return [
            {"symbol": s, "name": CSV_NAMES.get(s, s)}
            for s in self._symbols()
            if q in s.lower() or q in CSV_NAMES.get(s, "").lower()
        ][:limit]

    def company_name(self, symbol: str) -> str | None:
        return CSV_NAMES.get(symbol)


def get_provider(settings: Settings) -> DataProvider:
    if settings.data_provider == "csv":
        return CsvProvider(settings.sample_data_dir)
    if settings.data_provider == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unknown DATA_PROVIDER '{settings.data_provider}' (use 'yfinance' or 'csv').")
