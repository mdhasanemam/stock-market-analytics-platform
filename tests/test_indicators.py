import numpy as np
import pandas as pd

from backend.app.services import indicators as ind


def test_sma_matches_rolling_mean(prices):
    c = prices["Close"]
    assert np.allclose(ind.sma(c, 20).dropna(), c.rolling(20).mean().dropna())
    assert ind.sma(c, 20).iloc[:19].isna().all()


def test_ema_first_value_is_defined_after_span(prices):
    e = ind.ema(prices["Close"], 10)
    assert e.iloc[:9].isna().all() and not np.isnan(e.iloc[9])


def test_rsi_bounds_and_extremes(prices):
    r = ind.rsi(prices["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    rising = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(rising).iloc[-1] == 100.0
    falling = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(falling).iloc[-1] < 1.0


def test_macd_histogram_identity(prices):
    m = ind.macd(prices["Close"]).dropna()
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_bollinger_ordering(prices):
    b = ind.bollinger(prices["Close"]).dropna()
    assert (b["upper"] >= b["mid"]).all() and (b["mid"] >= b["lower"]).all()


def test_atr_positive(prices):
    assert (ind.atr(prices).dropna() > 0).all()


def test_obv_hand_computed():
    df = pd.DataFrame({"Close": [10, 11, 10, 10, 12], "Volume": [100, 200, 300, 400, 500]})
    assert ind.obv(df).tolist() == [0, 200, -100, -100, 400]


def test_indicators_do_not_use_future_data(prices):
    t = 700
    base = ind.compute_all(prices)
    changed = prices.copy()
    changed.iloc[t + 1:, :] = changed.iloc[t + 1:, :] * 3
    alt = ind.compute_all(changed)
    for key in ("sma20", "ema20", "rsi", "atr", "obv"):
        assert np.allclose(base[key].iloc[: t + 1].fillna(0), alt[key].iloc[: t + 1].fillna(0)), key
    assert np.allclose(base["macd"].iloc[: t + 1].fillna(0), alt["macd"].iloc[: t + 1].fillna(0))
