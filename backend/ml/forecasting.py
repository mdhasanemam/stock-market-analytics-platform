"""Future forecasts from a direct h-step model trained only on outcomes that are already known."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.core.errors import InsufficientData
from backend.ml.features import build_features, make_target
from backend.ml.models import get_model

MIN_POINTS_FOR_INTERVAL = 60


def forecast_future(df: pd.DataFrame, model_name: str, horizon: int, train_window: int,
                    residuals: np.ndarray | None, coverage: float = 0.90) -> dict:
    X = build_features(df)
    y = make_target(df["Close"], horizon)
    n = len(df)
    if X.iloc[-1].isna().any():
        raise InsufficientData("Latest feature row is incomplete; cannot forecast.")
    train_hi = n - 1 - horizon               # last row whose h-day outcome is already observed
    train_lo = max(0, train_hi - train_window + 1)
    Xtr, ytr = X.iloc[train_lo:train_hi + 1], y.iloc[train_lo:train_hi + 1]
    mask = Xtr.notna().all(axis=1) & ytr.notna()
    if mask.sum() < 100:
        raise InsufficientData("Too few clean rows to train the final model.")
    model = get_model(model_name).fit(Xtr[mask].to_numpy(), ytr[mask].to_numpy())
    pred_ret = float(model.predict(X.iloc[[-1]].to_numpy())[0])

    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1]
    # NOTE: business days only; exchange holidays are not modelled, so dates are approximate.
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    steps = np.arange(1, horizon + 1)
    # Direct model gives the h-day point forecast. Intermediate points are a straight line in log-price
    # (illustrative path, not separate model outputs).
    path = last_close * np.exp(pred_ret * steps / horizon)

    lower = upper = None
    interval = {"supported": False, "reason": "Fewer than %d out-of-sample residuals available." % MIN_POINTS_FOR_INTERVAL}
    if residuals is not None and len(residuals) >= MIN_POINTS_FOR_INTERVAL:
        lo_q, hi_q = np.quantile(residuals, [(1 - coverage) / 2, 1 - (1 - coverage) / 2])
        scale = np.sqrt(steps / horizon)  # random-walk style widening toward the horizon
        lower = last_close * np.exp(pred_ret * steps / horizon + lo_q * scale)
        upper = last_close * np.exp(pred_ret * steps / horizon + hi_q * scale)
        interval = {"supported": True, "coverage": coverage, "method":
                    "Empirical quantiles of walk-forward backtest residuals (log-return); approximate - "
                    "overlapping windows are autocorrelated for horizons > 1 and the future may differ from the past."}
    return {
        "kind": "future",
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "path": path.round(4).tolist(),
        "lower": None if lower is None else lower.round(4).tolist(),
        "upper": None if upper is None else upper.round(4).tolist(),
        "point": float(path[-1]),
        "predicted_return_pct": float((np.exp(pred_ret) - 1) * 100),
        "horizon_date": dates[-1].strftime("%Y-%m-%d"),
        "interval": interval,
        "trained_through": str(df.index[train_hi].date()),
    }
