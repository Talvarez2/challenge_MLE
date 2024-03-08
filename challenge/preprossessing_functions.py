from datetime import datetime
from challenge.config import (
    threshold_in_minutes,
    high_season_ranges,
    periods_of_the_day,
)


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
