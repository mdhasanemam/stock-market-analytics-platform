"""Validated, cached access to price history.

* validates/cleans provider output (never fabricates or interpolates prices)
* weekends/holidays are simply absent rows - they are NOT treated as missing data
* two-level cache (memory + CSV on disk); if the provider fails, stale cache is served with a warning
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from backend.app.core.config import Settings
from backend.app.core.errors import ProviderUnavailable, SymbolNotFound
from backend.app.services.providers import COLUMNS, DataProvider

log = logging.getLogger(__name__)
SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")
RANGES = {"1D": 1, "1W": 7, "1M": 30, "3M": 91, "6M": 182, "1Y": 365, "5Y": 365 * 5}


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not SYMBOL_RE.match(s):
        raise ValueError("Invalid ticker symbol.")
    return s


@dataclass
class PriceData:
    symbol: str
    df: pd.DataFrame
    source: str
    fetched_at: datetime
    stale: bool = False
    warnings: list[str] = field(default_factory=list)


def validate_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a cleaned frame plus human-readable warnings. Rows are dropped, never invented."""
    warnings: list[str] = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Provider data is missing columns: {missing}")
    out = df[COLUMNS].copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")

    bad = out["Close"].isna() | (out[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if bad.any():
        warnings.append(f"Dropped {int(bad.sum())} row(s) with missing or non-positive prices.")
        out = out[~bad]
    inconsistent = (out["High"] < out["Low"]) | (out["Close"] > out["High"] * 1.0001) | (out["Close"] < out["Low"] * 0.9999)
    if inconsistent.any():
        warnings.append(f"{int(inconsistent.sum())} row(s) have Close outside the High/Low range (kept; likely adjustment rounding).")
    out["Volume"] = out["Volume"].fillna(0)
    # Report unusually long gaps (> 5 business days). Normal weekends/holidays are not flagged.
    if len(out) > 1:
        gaps = out.index.to_series().diff().dt.days
        long_gaps = int((gaps > 7).sum())
        if long_gaps:
            warnings.append(f"{long_gaps} gap(s) longer than 7 calendar days in the history (halts or provider gaps).")
    return out, warnings


class DataService:
    def __init__(self, provider: DataProvider, settings: Settings):
        self.provider = provider
        self.settings = settings
        self._mem: dict[str, PriceData] = {}
        self._lock = threading.Lock()
        settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- cache helpers -------------------------------------------------
    def _disk_path(self, symbol: str):
        return self.settings.cache_dir / f"{self.provider.name}_{symbol}.csv"

    def _fresh(self, fetched_at: datetime) -> bool:
        age = datetime.now(timezone.utc) - fetched_at
        return age < timedelta(minutes=self.settings.cache_ttl_minutes)

    def _load_disk(self, symbol: str) -> PriceData | None:
        p = self._disk_path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, parse_dates=["Date"], index_col="Date")
            ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            return PriceData(symbol, df, self.provider.name, ts)
        except Exception as exc:
            log.warning("Corrupt cache for %s ignored: %s", symbol, exc)
            return None

    # --- public API ----------------------------------------------------
    def get(self, symbol: str) -> PriceData:
        symbol = normalize_symbol(symbol)
        with self._lock:
            cached = self._mem.get(symbol) or self._load_disk(symbol)
            if cached is not None and self._fresh(cached.fetched_at):
                self._mem[symbol] = cached
                return cached
            try:
                raw = self.provider.history(symbol)
                df, warns = validate_ohlcv(raw)
                if df.empty:
                    raise SymbolNotFound(f"No usable price rows for '{symbol}'.")
                pd_obj = PriceData(symbol, df, self.provider.name, datetime.now(timezone.utc), False, warns)
                self._mem[symbol] = pd_obj
                if self.provider.name != "csv":  # sample data is already local; no need to duplicate
                    df.to_csv(self._disk_path(symbol), index_label="Date")
                return pd_obj
            except ProviderUnavailable as exc:
                if cached is not None:
                    log.warning("Serving stale cache for %s: %s", symbol, exc)
                    cached.stale = True
                    cached.warnings = [*cached.warnings, f"Provider unavailable; showing cached data from {cached.fetched_at:%Y-%m-%d %H:%M} UTC."]
                    self._mem[symbol] = cached
                    return cached
                raise

    def slice(self, data: PriceData, range_key: str | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        df = data.df
        if range_key:
            key = range_key.upper()
            if key == "MAX":
                return df
            if key not in RANGES:
                raise ValueError(f"Unknown range '{range_key}'. Use one of {list(RANGES)} or MAX.")
            # ranges are anchored on the latest available bar, so they also work on historical sample data
            cutoff = df.index[-1] - timedelta(days=RANGES[key])
            out = df[df.index > cutoff]
            if key == "1D":  # last two bars so a daily change can be shown
                out = df.tail(2)
            return out
        if start or end:
            s = pd.to_datetime(start) if start else None
            e = pd.to_datetime(end) if end else None
            if s is not None and e is not None and s > e:
                raise ValueError("start must be on or before end.")
            return df.loc[s:e]
        return df

    def quote(self, symbol: str) -> dict:
        d = self.get(symbol)
        df = d.df
        last = df.iloc[-1]
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else float(last["Close"])
        change = float(last["Close"]) - prev_close
        return {
            "symbol": d.symbol,
            "name": self.provider.company_name(d.symbol),
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
            "open": float(last["Open"]), "high": float(last["High"]), "low": float(last["Low"]),
            "close": float(last["Close"]), "volume": int(last["Volume"]),
            "previous_close": prev_close,
            "change": change,
            "change_pct": (change / prev_close * 100) if prev_close else 0.0,
            "source": d.source, "stale": d.stale, "warnings": d.warnings,
        }
