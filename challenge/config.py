"""Configuration constants for the flight delay prediction model."""

PERIODS_OF_THE_DAY: dict[str, list[str]] = {
    "mañana": ["05:00", "11:59"],
    "tarde": ["12:00", "18:59"],
    "noche": ["19:00", "04:59"],
}

HIGH_SEASON_RANGES: list[tuple[str, str]] = [
    ("15-Dec", "31-Dec"),
    ("1-Jan", "3-Mar"),
    ("15-Jul", "31-Jul"),
    ("11-Sep", "30-Sep"),
]

THRESHOLD_IN_MINUTES: int = 15

TOP_10_FEATURES: list[str] = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]
