"""FastAPI application for flight delay prediction."""

import pickle
from typing import Any

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel, validator

from challenge.preprocessing import preprocess

VALID_TIPOVUELO = {"N", "I"}
VALID_MES_RANGE = range(1, 13)


class Flight(BaseModel):
    """Schema for a single flight prediction request."""

    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("TIPOVUELO")
    def _validate_tipovuelo(cls, v: str) -> str:
        if v not in VALID_TIPOVUELO:
            raise ValueError(f"TIPOVUELO must be one of {VALID_TIPOVUELO}, got '{v}'")
        return v

    @validator("MES")
    def _validate_mes(cls, v: int) -> int:
        if v not in VALID_MES_RANGE:
            raise ValueError(f"MES must be between 1 and 12, got {v}")
        return v


class PredictRequest(BaseModel):
    """Schema for the /predict endpoint request body."""

    flights: list[Flight]


class PredictResponse(BaseModel):
    """Schema for the /predict endpoint response."""

    predict: list[int]


def _load_model() -> Any:
    with open("./data/model.pkl", "rb") as f:
        return pickle.load(f)


model = _load_model()

app = fastapi.FastAPI(title="Flight Delay Prediction API")
handler = Mangum(app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 400 instead of 422 for validation errors to match test expectations."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Health check endpoint."""
    return {"status": "OK"}


@app.post("/predict", status_code=200, response_model=PredictResponse)
async def post_predict(data: PredictRequest) -> PredictResponse:
    """Predict flight delays.

    Args:
        data: Request body containing a list of flights.

    Returns:
        Predicted delay labels for each flight.
    """
    try:
        flights_df = pd.DataFrame([f.dict() for f in data.flights])
        features = preprocess(flights_df)
        prediction = model.predict(features)
        return PredictResponse(predict=prediction.tolist())
    except Exception as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
