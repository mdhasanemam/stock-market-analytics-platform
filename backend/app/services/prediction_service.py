"""Orchestrates data -> walk-forward evaluation -> forecast, with result caching."""
from __future__ import annotations

import logging
import threading
from datetime import timedelta

import numpy as np
import pandas as pd

from backend.app.core.config import Settings
from backend.app.core.errors import InsufficientData
from backend.app.services.data_service import DataService
from backend.ml.forecasting import forecast_future
from backend.ml.models import MODELS
from backend.ml.validation import walk_forward

log = logging.getLogger(__name__)
PERIOD_DAYS = {"2Y": 365 * 2, "5Y": 365 * 5, "10Y": 365 * 10, "MAX": None}
DISCLAIMER = ("Educational/research output, not investment advice. Forecasts are statistical estimates, "
              "not guaranteed future prices; markets are noisy and past performance does not predict future results.")


class PredictionService:
    def __init__(self, data: DataService, settings: Settings):
        self.data, self.settings = data, settings
        self._cache: dict[tuple, object] = {}
        self._lock = threading.Lock()

    def _history(self, symbol: str, period: str):
        period = period.upper()
        if period not in PERIOD_DAYS:
            raise ValueError(f"Unknown period '{period}'. Use one of {list(PERIOD_DAYS)}.")
        pdata = self.data.get(symbol)
        df = pdata.df
        if PERIOD_DAYS[period]:
            df = df[df.index > df.index[-1] - timedelta(days=PERIOD_DAYS[period])]
        return pdata, df, period

    def _wf(self, symbol: str, df: pd.DataFrame, model: str, horizon: int, period: str):
        key = (symbol, str(df.index[-1].date()), len(df), model, horizon, period)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        res = walk_forward(df, model, horizon, self.settings.ml_max_test_days, self.settings.ml_refit_every, self.settings.ml_train_window)
        with self._lock:
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[key] = res
        return res

    @staticmethod
    def _verdict(m: dict, kind: str) -> str:
        if kind == "baseline":
            return "Baseline reference model."
        vp, vd = m.get("mae_improvement_vs_persistence"), m.get("mae_improvement_vs_drift")
        if vp is None or vd is None:
            return "Could not compare with the naive baselines."
        if vp > 0.02 and vd > 0.02:
            return (f"In the out-of-sample backtest this model's MAE was {vp*100:.1f}% lower than 'price stays the same' and "
                    f"{vd*100:.1f}% lower than 'price drifts at its historical average'. Historical evidence only; it may not persist.")
        if vp > 0.02:
            return (f"This model beat 'price stays the same' by {vp*100:.1f}% MAE but NOT the simple historical-drift baseline "
                    f"({vd*100:+.1f}%). Much of its apparent skill is just the market's average upward drift in the test window.")
        return ("This model did NOT clearly beat the naive baselines in the out-of-sample backtest "
                f"(MAE vs persistence {vp*100:+.1f}%, vs drift {vd*100:+.1f}%). Treat its forecast with extra scepticism.")

    def predict(self, symbol: str, model: str, horizon: int, period: str = "5Y") -> dict:
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'.")
        pdata, df, period = self._history(symbol, period)
        res = self._wf(pdata.symbol, df, model, horizon, period)
        p = res.predictions
        drift_mae = self._wf(pdata.symbol, df, "drift", horizon, period).metrics["mae"]
        res.metrics["drift_mae"] = drift_mae
        res.metrics["mae_improvement_vs_drift"] = None if not drift_mae else float(1 - res.metrics["mae"] / drift_mae)
        resid = (p["actual_ret"] - p["pred_ret"]).to_numpy()
        fc = forecast_future(df, model, horizon, self.settings.ml_train_window, resid)
        info = MODELS[model]
        return {
            "symbol": pdata.symbol, "model": {"id": model, "label": info["label"], "kind": info["kind"], "description": info["description"]},
            "horizon": horizon, "period": period, "as_of": str(df.index[-1].date()), "last_close": float(df["Close"].iloc[-1]),
            "data_source": pdata.source, "stale": pdata.stale, "warnings": pdata.warnings,
            "history": {"dates": [d.strftime("%Y-%m-%d") for d in df.index[-260:]], "close": df["Close"].iloc[-260:].round(4).tolist()},
            "backtest": {
                "kind": "out_of_sample",
                "note": "Each point was predicted using only data available at its decision date; the model was never trained on the outcome it is scored against.",
                "dates": [d.strftime("%Y-%m-%d") for d in p["target_date"]],
                "actual": p["actual_price"].round(4).tolist(),
                "predicted": p["pred_price"].round(4).tolist(),
            },
            "forecast": fc,
            "metrics": res.metrics,
            "verdict": self._verdict(res.metrics, info["kind"]),
            "config": res.config,
            "disclaimer": DISCLAIMER,
        }

    def compare(self, symbol: str, models: list[str], horizon: int, period: str = "5Y") -> dict:
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}")
        models = list(dict.fromkeys(["persistence", "drift", *models]))  # baselines are always shown
        pdata, df, period = self._history(symbol, period)
        rows, series, dates, actual = [], {}, None, None
        for mid in models:
            try:
                res = self._wf(pdata.symbol, df, mid, horizon, period)
            except InsufficientData:
                raise
            except Exception as exc:  # one failing model should not sink the whole comparison
                log.exception("Model %s failed for %s", mid, symbol)
                rows.append({"id": mid, "label": MODELS[mid]["label"], "kind": MODELS[mid]["kind"], "error": str(exc)})
                continue
            p = res.predictions
            if dates is None:
                dates, actual = [d.strftime("%Y-%m-%d") for d in p["target_date"]], p["actual_price"].round(4).tolist()
            series[mid] = p["pred_price"].round(4).tolist()
            rows.append({"id": mid, "label": MODELS[mid]["label"], "kind": MODELS[mid]["kind"], "metrics": res.metrics})
        ok = [r for r in rows if "metrics" in r]
        drift_mae = next((r["metrics"]["mae"] for r in ok if r["id"] == "drift"), None)
        for r in ok:
            m = r["metrics"]
            m["drift_mae"] = drift_mae
            m["mae_improvement_vs_drift"] = None if not drift_mae else float(1 - m["mae"] / drift_mae)
            vp, vd = m["mae_improvement_vs_persistence"] or 0, m["mae_improvement_vs_drift"] or 0
            r["beats_persistence"] = bool(vp > 0.02) if r["kind"] == "ml" else None
            r["beats_all_baselines"] = bool(vp > 0.02 and vd > 0.02) if r["kind"] == "ml" else None
        for key, reverse in (("mae", False), ("rmse", False)):
            for rank, r in enumerate(sorted(ok, key=lambda r: r["metrics"][key]), 1):
                r[f"rank_{key}"] = rank
        leaderboard = sorted(ok, key=lambda r: r["metrics"]["mae"])
        ml_ok = [r for r in leaderboard if r["kind"] == "ml"]
        best = ml_ok[0] if ml_ok else None
        if not best:
            summary = "No machine-learning model was evaluated."
        elif any(r["beats_all_baselines"] for r in ml_ok):
            winners = [r["label"] for r in ml_ok if r["beats_all_baselines"]]
            summary = (f"Best ML model by MAE: {best['label']}. Models beating BOTH the persistence and historical-drift baselines "
                       f"by more than 2% MAE: {', '.join(winners)}. Historical backtest only - not a guarantee.")
        else:
            summary = (f"Best ML model by MAE: {best['label']}, but no ML model clearly beat both naive baselines "
                       "(persistence and historical drift) - common for daily price data. A high R-squared on prices is not evidence of skill.")
        return {
            "symbol": pdata.symbol, "horizon": horizon, "period": period, "as_of": str(df.index[-1].date()),
            "data_source": pdata.source, "n_test_points": ok[0]["metrics"]["n_test_points"] if ok else 0,
            "leaderboard": leaderboard, "failed": [r for r in rows if "error" in r],
            "best_ml_model": None if not best else best["id"],
            "summary": summary,
            "series": {"dates": dates, "actual": actual, "predicted": series},
            "disclaimer": DISCLAIMER,
        }
