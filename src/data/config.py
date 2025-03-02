import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.data.metadata import Coordinates, Station


@dataclass
class Config:
    """Configuration for prediction.

    Attributes:
        catalog: The catalog containing the raw metadata.
        stations: Stations with their coordinates.
        target_datetimes: Datetimes at which we want to do predictions.
        target_time_offsets: ???
    """

    catalog: pd.DataFrame
    stations: Dict[Station, Coordinates]
    target_datetimes: List[datetime]
    target_time_offsets: List[str]


def read_configuration_file(filename):
    """Read the configuration file as specified in the evaluation guidelines.

    Args:
        filename: path to the json config file.

    Returns:
        Config: the configuration object.
    """
    config_file = _load_config(filename)
    catalog = _load_catalog(config_file["dataframe_path"])
    stations = _load_stations(config_file["stations"])
    target_datetimes = _parse_datetime(config_file["target_datetimes"])

    return Config(
        catalog, stations, target_datetimes, config_file["target_time_offsets"],
    )


def _load_config(filename: str) -> Dict[str, Any]:
    with open(filename) as file:
        return json.load(file)


def _load_catalog(catalog_path: str) -> pd.DataFrame:
    with open(catalog_path, "rb") as file:
        return pickle.load(file)


def _load_stations(
    stations: Dict[str, Tuple[float, float, float]]
) -> Dict[Station, Coordinates]:
    parsed_station: Dict[str, Coordinates] = {}
    for name, coordinates in stations.items():
        parsed_station[Station(name)] = Coordinates(
            coordinates[0], coordinates[1], coordinates[2]
        )
    return parsed_station


def _parse_datetime(datetimes_str: List[str]) -> List[datetime]:
    return [datetime.fromisoformat(dt) for dt in datetimes_str]
