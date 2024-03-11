import pandas as pd
import numpy as np
from challenge.config import threshold_in_minutes, top_10_features
from datetime import datetime
from challenge.config import (
    threshold_in_minutes,
    high_season_ranges,
    periods_of_the_day,
)
from typing import Union

def preprocess(data: pd.DataFrame, target_column: str = None
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    if "Fecha-I" in data.columns:
        data["period_day"] = data["Fecha-I"].apply(get_period_day)
        data["high_season"] = data["Fecha-I"].apply(is_high_season)
        data["min_diff"] = data.apply(get_min_diff, axis=1)
        data["delay"] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)

    features = pd.concat(
        [
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ],
        axis=1,
    )
    for feature in top_10_features:
        if feature not in features:
            features[feature] = 0

    features = features[top_10_features]

    if target_column:
        target = data[[target_column]]
        return features, target
    else:
        return features

def get_period_day(date):
    # there is a slight chancge here qith the funtion in the model, as it no longer skips minutes between periods
    period = "noche"  # if the date is not in any period, it is considered night (should never happend if config is right)
    for key, value in periods_of_the_day.items():
        if date >= value[0] and date <= value[1]:
            period = key
            break
    return period


def is_high_season(fecha):
    fecha_año = int(fecha.split("-")[0])
    fecha = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")

    is_high_season_flag = 0
    for range in high_season_ranges:
        range_min = datetime.strptime(range[0], "%d-%b").replace(year=fecha_año)
        range_max = datetime.strptime(range[1], "%d-%b").replace(year=fecha_año)
        if fecha >= range_min and fecha <= range_max:
            is_high_season_flag = 1
            break
    return is_high_season_flag


def get_min_diff(data):
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff
