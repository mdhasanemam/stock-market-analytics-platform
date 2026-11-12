from fastapi import Request

from backend.app.database.db import Database
from backend.app.services.data_service import DataService
from backend.app.services.prediction_service import PredictionService


def data_service(request: Request) -> DataService:
    return request.app.state.data


def prediction_service(request: Request) -> PredictionService:
    return request.app.state.predictions


def database(request: Request) -> Database:
    return request.app.state.db
