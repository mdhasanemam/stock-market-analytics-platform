from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.services.data_service import SYMBOL_RE

Period = Literal["2Y", "5Y", "10Y", "MAX"]


class _SymbolMixin(BaseModel):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def _valid_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not SYMBOL_RE.match(v):
            raise ValueError("Invalid ticker symbol")
        return v


class PredictRequest(_SymbolMixin):
    model: str = "bayesian_ridge"
    horizon: int = Field(5, ge=1, le=30, description="Forecast horizon in trading days")
    period: Period = "5Y"
    save: bool = True


class CompareRequest(_SymbolMixin):
    models: list[str] = Field(default_factory=lambda: ["knn", "random_forest", "bayesian_ridge", "decision_tree", "gradient_boosting", "svr"], max_length=12)
    horizon: int = Field(5, ge=1, le=30)
    period: Period = "5Y"


class WatchlistAdd(_SymbolMixin):
    pass
