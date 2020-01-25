import pickle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Generator, Optional

import pandas as pd


class Station(Enum):
    """All SURFRAD station names."""

    BND = "BND"
    TBL = "TBL"
    DRA = "DRA"
    FPK = "FPK"
    GWN = "GWN"
    PSU = "PSU"
    SXF = "SXF"


class UnableToLoadMetadata(Exception):
    """Exception that occured when something wrong happened when loading metadata."""

    pass


@dataclass
class Metadata:
    """Metadata containing the information necessary to train some models."""

    image_path: str
    datetime: datetime
    target: Optional[float]
    target_1h: Optional[float]
    target_3h: Optional[float]
    target_6h: Optional[float]


class MetadataLoader:
    """Load the metadata from the catalog for differents stations."""

    def __init__(self, file_name: str) -> None:
        """Create a metadata loader.

        :param file_name: Path to the catalog file.
            The file is supposed to be a pickle file
            containing a pandas dataframe.
        """
        self.file_name = file_name

    def load(
        self, station: Station, compression="8bit", night_time=True
    ) -> Generator[Metadata, None, None]:
        """Load the metadata from the catalog.

        :param station: The station which impact which target, target_1h,
            target_3h and target_6h that will be included in the metadata.
        :param compression: If the image_path point to a compressed image.
            Possible values are [None, 8bit, 16bit] (Default 8bit)
        :param night_time: If the night time must be included.
            The night time is calculated depending of the station.
        :return: A generator of metadata.
        """
        catalog = self._load_file()
        image_column = self._image_column(compression)

        catalog = self._filter_null(catalog, image_column)
        catalog = self._filter_null(catalog, f"{station.name}_GHI")
        catalog = self._filter_night(catalog, station, night_time)

        for index, row in catalog.iterrows():
            yield self._build_metadata(catalog, station, image_column, index, row)

    def _find_futur_target(
        self, catalog: pd.DataFrame, station: Station, time: pd.Timestamp, hour: int
    ) -> Optional[float]:
        index = time + pd.to_timedelta(hour, unit="h")

        try:
            return catalog.loc[index][f"{station.name}_GHI"]
        except KeyError:
            return None

    def _image_column(self, compression: str):
        if compression is None:
            return "ncdf_path"
        elif compression == "8bit":
            return "hdf5_8bit_path"
        elif compression == "16bit":
            return "hdf5_16bit_path"
        else:
            raise UnableToLoadMetadata(f"Unsupported compression: {compression}")

    def _filter_null(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df = df[df[column] != "nan"]
        df = df[df[column].notnull()]

        return df

    def _filter_night(
        self, df: pd.DataFrame, station: Station, night_time: bool
    ) -> pd.DataFrame:
        if not night_time:
            return df[df[f"{station.name}_DAYTIME"] == 1]

        return df

    def _build_metadata(
        self,
        catalog: pd.DataFrame,
        station: Station,
        image_column: str,
        timestamp: pd.Timestamp,
        row: pd.Series,
    ) -> Metadata:
        image_path = row[image_column]
        target = row[f"{station.name}_GHI"]

        target_1h = self._find_futur_target(catalog, station, timestamp, 1)
        target_3h = self._find_futur_target(catalog, station, timestamp, 3)
        target_6h = self._find_futur_target(catalog, station, timestamp, 6)

        datetime = timestamp.to_pydatetime()
        return Metadata(image_path, datetime, target, target_1h, target_3h, target_6h)

    def _load_file(self) -> pd.DataFrame:
        try:
            with open(self.file_name, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError as e:
            raise UnableToLoadMetadata(e)
