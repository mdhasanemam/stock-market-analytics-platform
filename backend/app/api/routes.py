from __future__ import annotations

import csv
import io
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.app.api.deps import data_service, database, prediction_service
from backend.app.database.db import Database
from backend.app.schemas.requests import CompareRequest, PredictRequest, WatchlistAdd
from backend.app.services import indicators as ind
from backend.app.services.data_service import DataService, normalize_symbol
from backend.app.services.prediction_service import PredictionService
from backend.ml.models import catalog

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

FEATURED = ["AAPL", "MSFT", "TSLA", "FB"]  # overridden at runtime for live providers


def _sym(symbol: str) -> str:
    try:
        return normalize_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _clean(series: pd.Series | np.ndarray, nd: int = 4) -> list:
    a = pd.Series(series).astype(float).round(nd)
    return [None if pd.isna(v) else float(v) for v in a]


@router.get("/health")
def health(ds: DataService = Depends(data_service)):
    return {"status": "ok", "data_provider": ds.provider.name, "provider_note": ds.provider.description}


@router.get("/stocks/search")
def search(q: str = Query(..., min_length=1, max_length=40), ds: DataService = Depends(data_service)):
    return {"results": ds.provider.search(q)}


@router.get("/overview")
def overview(ds: DataService = Depends(data_service)):
    syms = ["AAPL", "MSFT", "TSLA", "FB"] if ds.provider.name == "csv" else ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN"]
    cards, errors = [], []
    for s in syms:
        try:
            q = ds.quote(s)
            d = ds.get(s).df.tail(60)
            q["spark"] = _clean(d["Close"], 2)
            cards.append(q)
        except Exception as exc:  # one failing symbol must not break the home page
            errors.append({"symbol": s, "error": str(exc)})
    trend = []
    for c in cards:
        df = ds.get(c["symbol"]).df
        c50 = df["Close"].rolling(50).mean().iloc[-1]
        label = "Neutral" if pd.isna(c50) else ("Bullish" if c["close"] > c50 * 1.02 else ("Bearish" if c["close"] < c50 * 0.98 else "Neutral"))
        trend.append({"symbol": c["symbol"], "trend": label})
    return {"featured": cards, "trend": trend, "errors": errors, "data_source": ds.provider.name, "provider_note": ds.provider.description}


@router.get("/stocks/{symbol}/quote")
def quote(symbol: str, ds: DataService = Depends(data_service)):
    return ds.quote(_sym(symbol))


@router.get("/stocks/{symbol}/history")
def history(symbol: str, range: str | None = Query("1Y"), start: str | None = None, end: str | None = None,
            ds: DataService = Depends(data_service)):
    pdata = ds.get(_sym(symbol))
    try:
        df = ds.slice(pdata, None if (start or end) else range, start, end)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if df.empty:
        raise HTTPException(404, "No data in the requested range.")
    first, last = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
    return {
        "symbol": pdata.symbol, "source": pdata.source, "stale": pdata.stale, "warnings": pdata.warnings,
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "open": _clean(df["Open"]), "high": _clean(df["High"]), "low": _clean(df["Low"]),
        "close": _clean(df["Close"]), "volume": [int(v) for v in df["Volume"]],
        "summary": {"start": df.index[0].strftime("%Y-%m-%d"), "end": df.index[-1].strftime("%Y-%m-%d"), "n": len(df),
                    "change": last - first, "change_pct": (last / first - 1) * 100 if first else None,
                    "high": float(df["High"].max()), "low": float(df["Low"].min())},
    }


@router.get("/stocks/{symbol}/indicators")
def indicators(symbol: str, range: str | None = Query("1Y"), start: str | None = None, end: str | None = None,
               ds: DataService = Depends(data_service)):
    """Indicators are computed on the FULL history (so long windows are warmed up), then sliced to the range."""
    pdata = ds.get(_sym(symbol))
    full = ind.compute_all(pdata.df)
    try:
        sliced = ds.slice(pdata, None if (start or end) else range, start, end)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    idx = sliced.index
    out: dict = {"dates": [d.strftime("%Y-%m-%d") for d in idx], "info": ind.INDICATOR_INFO}
    for k, v in full.items():
        if isinstance(v, pd.DataFrame):
            out[k] = {c: _clean(v[c].reindex(idx)) for c in v.columns}
        else:
            out[k] = _clean(v.reindex(idx), 2 if k == "obv" else 4)
    return out


@router.get("/models")
def models():
    return {"models": catalog()}


@router.post("/predict")
def predict(req: PredictRequest, ps: PredictionService = Depends(prediction_service), db: Database = Depends(database)):
    result = ps.predict(req.symbol, req.model, req.horizon, req.period)
    result["saved_id"] = db.save_prediction(result) if req.save else None
    return result


@router.post("/compare")
def compare(req: CompareRequest, ps: PredictionService = Depends(prediction_service)):
    return ps.compare(req.symbol, req.models, req.horizon, req.period)


# ---------------------------------------------------------------- watchlist
@router.get("/watchlist")
def get_watchlist(ds: DataService = Depends(data_service), db: Database = Depends(database)):
    items = []
    for s in db.watchlist():
        try:
            q = ds.quote(s)
            q["spark"] = _clean(ds.get(s).df["Close"].tail(30), 2)
            items.append(q)
        except Exception as exc:
            items.append({"symbol": s, "error": str(exc)})
    return {"items": items}


@router.post("/watchlist", status_code=201)
def add_watch(req: WatchlistAdd, ds: DataService = Depends(data_service), db: Database = Depends(database)):
    ds.get(req.symbol)  # reject symbols that have no data
    if not db.add_watch(req.symbol):
        raise HTTPException(409, f"{req.symbol} is already on the watchlist.")
    return {"symbol": req.symbol}


@router.delete("/watchlist/{symbol}")
def remove_watch(symbol: str, db: Database = Depends(database)):
    if not db.remove_watch(_sym(symbol)):
        raise HTTPException(404, "Symbol is not on the watchlist.")
    return {"removed": symbol.upper()}


# ---------------------------------------------------------------- prediction history
@router.get("/history")
def list_history(limit: int = Query(100, ge=1, le=500), db: Database = Depends(database)):
    return {"items": db.list_predictions(limit)}


@router.get("/history/{pid}")
def get_history(pid: int, db: Database = Depends(database)):
    rec = db.get_prediction(pid)
    if not rec:
        raise HTTPException(404, "Prediction not found.")
    return rec


@router.delete("/history/{pid}")
def delete_history(pid: int, db: Database = Depends(database)):
    if not db.delete_prediction(pid):
        raise HTTPException(404, "Prediction not found.")
    return {"deleted": pid}


@router.get("/history/{pid}/export.csv")
def export_csv(pid: int, db: Database = Depends(database)):
    rec = db.get_prediction(pid)
    if not rec:
        raise HTTPException(404, "Prediction not found.")
    fc = rec["forecast"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "forecast_price", "lower", "upper", "symbol", "model", "horizon_days", "as_of", "type"])
    for i, d in enumerate(fc["dates"]):
        w.writerow([d, fc["path"][i], (fc["lower"] or [None] * len(fc["dates"]))[i], (fc["upper"] or [None] * len(fc["dates"]))[i],
                    rec["symbol"], rec["model"], rec["horizon"], rec["as_of"], "future_forecast (not guaranteed)"])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="forecast_{rec["symbol"]}_{rec["id"]}.csv"'})
