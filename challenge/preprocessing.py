"""Preprocessing functions for flight delay prediction."""

from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from challenge.config import (
    HIGH_SEASON_RANGES,
    PERIODS_OF_THE_DAY,
    THRESHOLD_IN_MINUTES,
    TOP_10_FEATURES,
)


def preprocess(
    data: pd.DataFrame, target_column: str | None = None
) -> Union[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Prepare raw data for training or prediction.

    Args:
        data: Raw flight data.
        target_column: If set, the target column is returned alongside features.

    Returns:
        Features DataFrame, or a tuple of (features, target) when target_column is set.
    """
    if "Fecha-I" in data.columns:
        data["period_day"] = data["Fecha-I"].apply(_get_period_day)
        data["high_season"] = data["Fecha-I"].apply(_is_high_season)
        data["min_diff"] = data.apply(_get_min_diff, axis=1)
        data["delay"] = np.where(data["min_diff"] > THRESHOLD_IN_MINUTES, 1, 0)

    features = pd.concat(
        [
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ],
        axis=1,
    )

    for feature in TOP_10_FEATURES:
        if feature not in features.columns:
            features[feature] = 0

    features = features[TOP_10_FEATURES]

    if target_column:
        return features, data[[target_column]]
    return features


def _get_period_day(date: str) -> str:
    """Classify a datetime string into a period of the day."""
    date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
    for period, (start, end) in PERIODS_OF_THE_DAY.items():
        start_time = datetime.strptime(start, "%H:%M").time()
        end_time = datetime.strptime(end, "%H:%M").time()
        if start_time <= end_time:
            if start_time <= date_time <= end_time:
                return period
        else:  # overnight range (noche)
            if date_time >= start_time or date_time <= end_time:
                return period
    return "noche"


def _is_high_season(fecha: str) -> int:
    """Return 1 if the date falls within a high-season range, 0 otherwise."""
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
    year = fecha_dt.year

    for range_start, range_end in HIGH_SEASON_RANGES:
        dt_min = datetime.strptime(range_start, "%d-%b").replace(year=year)
        dt_max = datetime.strptime(range_end, "%d-%b").replace(year=year)
        if dt_min <= fecha_dt <= dt_max:
            return 1
    return 0


def _get_min_diff(row: pd.Series) -> float:
    """Calculate the difference in minutes between operation and scheduled times."""
    fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    return (fecha_o - fecha_i).total_seconds() / 60
