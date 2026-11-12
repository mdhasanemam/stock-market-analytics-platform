"""Chronological walk-forward (expanding/rolling-window) evaluation. Data is never shuffled.

For a decision made at row t with horizon h the outcome C[t+h] is only known at row t+h. Therefore a model
refit at row s may only train on rows i with i + h <= s ("purging" the last h rows) - otherwise targets
would peek past the decision date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.app.core.errors import InsufficientData
from backend.ml.features import build_features, make_target
from backend.ml.metrics import directional_accuracy, price_metrics, return_skill
from backend.ml.models import get_model

MIN_TRAIN_ROWS = 200
MIN_TEST_ROWS = 40


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame          # one row per out-of-sample decision date
    fits: list[dict]                   # [{train_start, train_end, test_start, test_end}] positional, for auditing
    metrics: dict
    config: dict


def _fit_predict(model_name: str, Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame) -> np.ndarray:
    mask = Xtr.notna().all(axis=1) & ytr.notna()
    if mask.sum() < 50:
        raise InsufficientData("Fewer than 50 clean training rows in a walk-forward window.")
    model = get_model(model_name).fit(Xtr[mask].to_numpy(), ytr[mask].to_numpy())
    return model.predict(Xte.to_numpy())


def plan_test_window(n_rows: int, first_valid: int, horizon: int, max_test_days: int) -> tuple[int, int]:
    """Return (start_t, last_t) positional indices of out-of-sample decision rows."""
    last_t = n_rows - 1 - horizon
    usable = last_t - first_valid - horizon - MIN_TRAIN_ROWS + 1
    test_days = min(max_test_days, usable // 3 if usable < max_test_days * 3 else max_test_days)
    if test_days < MIN_TEST_ROWS:
        raise InsufficientData(
            f"Not enough history: need at least ~{MIN_TRAIN_ROWS + 3 * MIN_TEST_ROWS + first_valid + horizon} trading days "
            f"for a {horizon}-day horizon, have {n_rows}. Choose a longer history period."
        )
    return last_t - test_days + 1, last_t


def walk_forward(df: pd.DataFrame, model_name: str, horizon: int, max_test_days: int = 250,
                 refit_every: int = 25, train_window: int = 1500) -> WalkForwardResult:
    X = build_features(df)
    y = make_target(df["Close"], horizon)
    n = len(df)
    ok = X.notna().all(axis=1).to_numpy()
    if not ok.any():
        raise InsufficientData("Could not compute features (history too short).")
    first_valid = int(np.argmax(ok))
    start_t, last_t = plan_test_window(n, first_valid, horizon, max_test_days)

    close = df["Close"].to_numpy()
    idx = df.index
    rows, fits = [], []
    s = start_t
    while s <= last_t:
        e = min(s + refit_every - 1, last_t)
        train_hi = s - horizon                      # inclusive; guarantees target known at decision row s
        train_lo = max(first_valid, train_hi - train_window + 1)
        pred = _fit_predict(model_name, X.iloc[train_lo:train_hi + 1], y.iloc[train_lo:train_hi + 1], X.iloc[s:e + 1])
        for k, t in enumerate(range(s, e + 1)):
            if not ok[t]:
                continue
            rows.append((idx[t], idx[t + horizon], close[t], close[t + horizon], pred[k]))
        fits.append({"train_start": train_lo, "train_end": train_hi, "test_start": s, "test_end": e})
        s = e + 1

    p = pd.DataFrame(rows, columns=["decision_date", "target_date", "close", "actual_price", "pred_ret"])
    p["pred_price"] = p["close"] * np.exp(p["pred_ret"])
    p["actual_ret"] = np.log(p["actual_price"] / p["close"])

    a, pr = p["actual_price"].to_numpy(), p["pred_price"].to_numpy()
    m = price_metrics(a, pr)
    m["directional_accuracy"] = directional_accuracy(p["actual_ret"].to_numpy(), p["pred_ret"].to_numpy())
    m["return_skill_vs_persistence"] = return_skill(p["actual_ret"].to_numpy(), p["pred_ret"].to_numpy())
    persist = price_metrics(a, p["close"].to_numpy())
    m["persistence_mae"] = persist["mae"]
    m["mae_improvement_vs_persistence"] = None if not persist["mae"] else float(1 - m["mae"] / persist["mae"])
    m["n_test_points"] = int(len(p))
    cfg = {"horizon": horizon, "max_test_days": max_test_days, "refit_every": refit_every, "train_window": train_window,
           "target": "log-return over horizon", "n_refits": len(fits),
           "test_start": str(p["decision_date"].iloc[0].date()), "test_end": str(p["decision_date"].iloc[-1].date())}
    return WalkForwardResult(p, fits, m, cfg)
