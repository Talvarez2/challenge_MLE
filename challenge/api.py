"""FastAPI application for flight delay prediction."""

import logging

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.config import MODEL_PATH
from challenge.model import DelayModel
from challenge.preprocessing import preprocess

logger = logging.getLogger(__name__)

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


try:
    model = DelayModel.load_model(MODEL_PATH)
except FileNotFoundError:
    logger.warning("Model file not found at %s; predictions will return zeros.", MODEL_PATH)
    model = DelayModel()

app = fastapi.FastAPI(title="Flight Delay Prediction API")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 400 instead of 422 for validation errors to match test expectations."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", status_code=200)
async def get_health() -> dict[str, str]:
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
    flights_df = pd.DataFrame([f.dict() for f in data.flights])
    features = preprocess(flights_df)
    prediction = model.predict(features)
    return PredictResponse(predict=prediction)
