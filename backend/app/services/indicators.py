"""Technical indicators (pure pandas, no look-ahead: value at t uses data up to and including t)."""
from __future__ import annotations

import pandas as pd

INDICATOR_INFO = {
    "sma": {"name": "Simple Moving Average", "panel": "overlay",
            "description": "Average closing price over N days. Smooths noise; price above a rising SMA is commonly read as an uptrend. Lagging by construction."},
    "ema": {"name": "Exponential Moving Average", "panel": "overlay",
            "description": "Like the SMA but weights recent prices more heavily, so it reacts faster to changes."},
    "bollinger": {"name": "Bollinger Bands", "panel": "overlay",
                  "description": "SMA(20) plus/minus 2 standard deviations. Band width reflects recent volatility; touching a band is not by itself a signal."},
    "rsi": {"name": "Relative Strength Index", "panel": "sub",
            "description": "Momentum oscillator (0-100, Wilder smoothing, 14 days). Above 70 is often called overbought and below 30 oversold, but strong trends can stay extreme for long."},
    "macd": {"name": "MACD", "panel": "sub",
             "description": "EMA(12) - EMA(26), with a 9-day EMA signal line and histogram. Crossovers are used to describe momentum shifts."},
    "atr": {"name": "Average True Range", "panel": "sub",
            "description": "Wilder-smoothed average of the true range over 14 days. Measures volatility in price units, not direction."},
    "obv": {"name": "On-Balance Volume", "panel": "sub",
            "description": "Cumulative volume that adds on up days and subtracts on down days. Divergence between OBV and price is sometimes used to question a trend."},
}


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int = 20) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out[(loss == 0) & (gain > 0)] = 100.0
    out[(loss == 0) & (gain == 0)] = 50.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False, min_periods=fast).mean() - close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["High"].iloc[0] - df["Low"].iloc[0]
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = df["Close"].diff().apply(lambda x: 0 if pd.isna(x) else (1 if x > 0 else (-1 if x < 0 else 0)))
    return (direction * df["Volume"]).cumsum()


def compute_all(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    c = df["Close"]
    return {
        "sma20": sma(c, 20), "sma50": sma(c, 50), "sma200": sma(c, 200),
        "ema20": ema(c, 20), "ema50": ema(c, 50),
        "bollinger": bollinger(c), "rsi": rsi(c), "macd": macd(c),
        "atr": atr(df), "obv": obv(df),
    }
