import numpy as np
import pytest

from backend.app.core.errors import InsufficientData
from backend.ml.features import FEATURE_COLUMNS, build_features, make_target
from backend.ml.forecasting import forecast_future
from backend.ml.models import MODELS, get_model
from backend.ml.validation import walk_forward


def test_features_never_use_future_rows(prices):
    t = 800
    base = build_features(prices)
    changed = prices.copy()
    changed.iloc[t + 1:, :] = changed.iloc[t + 1:, :] * 5
    alt = build_features(changed)
    assert np.allclose(base.iloc[: t + 1].fillna(-9), alt.iloc[: t + 1].fillna(-9))


def test_target_is_future_log_return(prices):
    y = make_target(prices["Close"], 5)
    assert np.isclose(y.iloc[10], np.log(prices["Close"].iloc[15] / prices["Close"].iloc[10]))
    assert y.iloc[-5:].isna().all()


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_walk_forward_is_chronological_and_purged(prices, horizon):
    res = walk_forward(prices, "bayesian_ridge", horizon, max_test_days=120, refit_every=20, train_window=500)
    fits = res.fits
    for f in fits:
        assert f["train_end"] + horizon <= f["test_start"], "training targets must be known at the decision date"
        assert f["train_end"] < f["test_start"]
        assert f["train_start"] <= f["train_end"]
    for a, b in zip(fits, fits[1:]):
        assert b["test_start"] == a["test_end"] + 1          # contiguous, no overlap, no shuffling
    p = res.predictions
    assert p["decision_date"].is_monotonic_increasing
    assert (p["target_date"] > p["decision_date"]).all()


def test_walk_forward_predictions_ignore_the_future(prices):
    """Corrupting prices far after the first block must not change that block's predictions."""
    h = 5
    kw = dict(horizon=h, max_test_days=100, refit_every=20, train_window=500)
    base = walk_forward(prices, "bayesian_ridge", **kw)
    first = base.fits[0]
    cut = first["test_end"] + 1                      # everything from here on is 'future' to block 1
    changed = prices.copy()
    changed.iloc[cut:, :] = changed.iloc[cut:, :] * 7
    alt = walk_forward(changed, "bayesian_ridge", **kw)
    n_block = first["test_end"] - first["test_start"] + 1
    assert np.allclose(base.predictions["pred_ret"].iloc[:n_block], alt.predictions["pred_ret"].iloc[:n_block])


def test_metrics_and_persistence_baseline(prices):
    res = walk_forward(prices, "persistence", 5, max_test_days=100, refit_every=25, train_window=500)
    p = res.predictions
    assert np.allclose(p["pred_price"], p["close"])
    assert res.metrics["mae_improvement_vs_persistence"] == pytest.approx(0.0)
    assert res.metrics["directional_accuracy"] is None       # zero forecasts carry no direction
    for k in ("mae", "mse", "rmse", "r2", "mape", "n_test_points"):
        assert k in res.metrics


def test_all_models_run_and_return_finite_predictions(prices):
    X = build_features(prices).dropna()
    y = make_target(prices["Close"], 5).reindex(X.index).dropna()
    X = X.loc[y.index].iloc[:400]
    y = y.iloc[:400]
    for name in MODELS:
        m = get_model(name).fit(X.to_numpy(), y.to_numpy())
        pred = m.predict(X.to_numpy()[:5])
        assert pred.shape == (5,) and np.isfinite(pred).all(), name
    assert X.shape[1] == len(FEATURE_COLUMNS)


def test_insufficient_history_is_reported(prices):
    with pytest.raises(InsufficientData):
        walk_forward(prices.iloc[:150], "bayesian_ridge", 5)


def test_forecast_shape_dates_and_interval(prices):
    res = walk_forward(prices, "bayesian_ridge", 5, max_test_days=120, refit_every=20, train_window=500)
    resid = (res.predictions["actual_ret"] - res.predictions["pred_ret"]).to_numpy()
    fc = forecast_future(prices, "bayesian_ridge", 5, 500, resid)
    assert len(fc["dates"]) == len(fc["path"]) == 5
    assert fc["dates"][0] > str(prices.index[-1].date())
    assert fc["interval"]["supported"]
    assert all(lo <= p <= hi for lo, p, hi in zip(fc["lower"], fc["path"], fc["upper"]))
    assert fc["trained_through"] < str(prices.index[-1].date())
    none = forecast_future(prices, "bayesian_ridge", 5, 500, resid[:10])
    assert none["interval"]["supported"] is False and none["lower"] is None
