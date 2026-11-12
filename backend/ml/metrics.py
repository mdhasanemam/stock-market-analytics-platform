from __future__ import annotations

import numpy as np


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def price_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    nonzero = actual != 0
    return {
        "mae": _f(np.mean(np.abs(err))),
        "mse": _f(mse),
        "rmse": _f(np.sqrt(mse)),
        "r2": _f(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None,
        "mape": _f(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
    }


def directional_accuracy(actual_ret: np.ndarray, pred_ret: np.ndarray) -> float | None:
    """Share of forecasts whose predicted direction matches the realised direction.
    Forecasts of exactly zero (the persistence baseline) carry no direction and are excluded."""
    mask = pred_ret != 0
    if not mask.any():
        return None
    return _f(np.mean(np.sign(actual_ret[mask]) == np.sign(pred_ret[mask])))


def return_skill(actual_ret: np.ndarray, pred_ret: np.ndarray) -> float | None:
    """Out-of-sample R^2 of the return forecast relative to a zero-return forecast (Campbell-Thompson style).
    > 0 means better than persistence in squared-error terms."""
    denom = np.sum(actual_ret ** 2)
    return _f(1 - np.sum((actual_ret - pred_ret) ** 2) / denom) if denom > 0 else None
