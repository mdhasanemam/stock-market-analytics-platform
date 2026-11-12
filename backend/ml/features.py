"""Leak-free feature engineering.

Every feature at row t is computed from information available at the close of day t.
The target for horizon h is the future log-return log(C[t+h] / C[t]); it is built separately
and must never be used as a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.services import indicators as ind

FEATURE_COLUMNS = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_20", "dev_sma10", "dev_sma20", "dev_sma50", "dev_ema20",
    "rsi_14", "macd_hist_pct", "bb_pctb", "atr_pct", "volume_z", "range_pct", "gap_pct",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["Close"], df["Volume"]
    lc = np.log(c)
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_{k}"] = lc.diff(k)
    r1 = lc.diff()
    f["vol_10"] = r1.rolling(10).std()
    f["vol_20"] = r1.rolling(20).std()
    for w in (10, 20, 50):
        f[f"dev_sma{w}"] = np.log(c / ind.sma(c, w))
    f["dev_ema20"] = np.log(c / ind.ema(c, 20))
    f["rsi_14"] = ind.rsi(c) / 100.0
    f["macd_hist_pct"] = ind.macd(c)["hist"] / c
    bb = ind.bollinger(c)
    f["bb_pctb"] = (c - bb["lower"]) / (bb["upper"] - bb["lower"]).replace(0, np.nan)
    f["atr_pct"] = ind.atr(df) / c
    lv = np.log1p(v)
    f["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std().replace(0, np.nan)
    f["range_pct"] = (df["High"] - df["Low"]) / c
    f["gap_pct"] = np.log(df["Open"] / c.shift(1))
    return f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series, horizon: int) -> pd.Series:
    """log(C[t+h] / C[t]) - NaN for the final `horizon` rows (future not yet observed)."""
    lc = np.log(close)
    return (lc.shift(-horizon) - lc).rename(f"target_h{horizon}")
