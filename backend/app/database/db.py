"""SQLite persistence (SQLAlchemy 2.0). Chosen over MongoDB: the data is small, relational and single-node;
swap DATABASE_URL for PostgreSQL in production without code changes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class WatchlistItem(Base):
    __tablename__ = "watchlist"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    model: Mapped[str] = mapped_column(String(40))
    horizon: Mapped[int] = mapped_column(Integer)
    period: Mapped[str] = mapped_column(String(8))
    as_of: Mapped[str] = mapped_column(String(10))            # last market date used
    forecast_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
    config_json: Mapped[str] = mapped_column(Text)


class Database:
    def __init__(self, url: str):
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
            from pathlib import Path
            Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, **kwargs)
        if url == "sqlite:///:memory:":
            from sqlalchemy.pool import StaticPool
            self.engine = create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._session()

    # watchlist -----------------------------------------------------------
    def watchlist(self) -> list[str]:
        with self.session() as s:
            return [r.symbol for r in s.scalars(select(WatchlistItem).order_by(WatchlistItem.added_at))]

    def add_watch(self, symbol: str) -> bool:
        with self.session() as s:
            if s.scalar(select(WatchlistItem).where(WatchlistItem.symbol == symbol)):
                return False
            s.add(WatchlistItem(symbol=symbol))
            s.commit()
            return True

    def remove_watch(self, symbol: str) -> bool:
        with self.session() as s:
            row = s.scalar(select(WatchlistItem).where(WatchlistItem.symbol == symbol))
            if not row:
                return False
            s.delete(row)
            s.commit()
            return True

    # prediction history ----------------------------------------------------
    def save_prediction(self, result: dict) -> int:
        with self.session() as s:
            rec = PredictionRecord(
                symbol=result["symbol"], model=result["model"]["id"], horizon=result["horizon"], period=result["period"],
                as_of=result["as_of"],
                forecast_json=json.dumps(result["forecast"]), metrics_json=json.dumps(result["metrics"]),
                config_json=json.dumps({**result["config"], "model": result["model"], "data_source": result["data_source"]}),
            )
            s.add(rec)
            s.commit()
            return rec.id

    @staticmethod
    def _rec(r: PredictionRecord, full: bool) -> dict:
        d = {"id": r.id, "created_at": r.created_at.isoformat(), "symbol": r.symbol, "model": r.model, "horizon": r.horizon,
             "period": r.period, "as_of": r.as_of, "metrics": json.loads(r.metrics_json)}
        fc = json.loads(r.forecast_json)
        d["forecast_point"], d["forecast_date"] = fc.get("point"), fc.get("horizon_date")
        if full:
            d["forecast"], d["config"] = fc, json.loads(r.config_json)
        return d

    def list_predictions(self, limit: int = 100) -> list[dict]:
        with self.session() as s:
            return [self._rec(r, False) for r in s.scalars(select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit))]

    def get_prediction(self, pid: int) -> dict | None:
        with self.session() as s:
            r = s.get(PredictionRecord, pid)
            return self._rec(r, True) if r else None

    def delete_prediction(self, pid: int) -> bool:
        with self.session() as s:
            r = s.get(PredictionRecord, pid)
            if not r:
                return False
            s.delete(r)
            s.commit()
            return True
