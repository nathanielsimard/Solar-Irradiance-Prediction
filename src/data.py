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

    image_path: str  # full path on helios
    image_compression: str  # "8bit", "16bit" or "None"
    image_offset: int  # Indice de l'image dans le fichier hd5
    datetime: datetime  # UTC datetime
    latitude: float  # Coordinates of the station.
    longitude: float
    altitude: float
    target_ghi: Optional[float]  # GHI, non normalized, watts/m2
    target_ghi_1h: Optional[float]  # Same, T+1h
    target_ghi_3h: Optional[float]  # Same, T+3h
    target_ghi_6h: Optional[float]  # Same, T+6h
    # Cloudiness category. ("night", #"cloudy", "clear", "variable", "slightly cloudy")
    target_cloudiness: Optional[str]
    target_cloudiness_1h: Optional[str]  # Same, T+1h
    target_cloudiness_3h: Optional[str]  # Same, T+3h
    target_cloudiness_6h: Optional[str]  # Same, T+6h


class MetadataLoader:
    """Load the metadata from the catalog for differents stations."""

    # Dictionnary of all SURFRAD station names and locations
    # Given as (latitude, longitude, altitude) tuples

    Stations = {
        "BND": [40.05192, -88.37309, 230],
        "TBL": [40.12498, -105.23680, 1689],
        "DRA": [36.62373, -116.01947, 1007],
        "FPK": [48.30783, -105.10170, 634],
        "GWN": [34.25470, -89.87290, 98],
        "PSU": [40.72012, -77.93085, 376],
        "SXF": [43.73403, -96.62328, 473],
    }

    def __init__(self, file_name: str = None, dataframe=None) -> None:
        """Create a metadata loader.

        :param file_name: Path to the catalog file.
            The file is supposed to be a pickle file
            containing a pandas dataframe.

        :param dataframe: The pandas datafram as loader by the dataloader. Can be used instead
            of the path to the actual dataframe when required.

            Those parameters are mutually exclusive and should not be provided at the same time.

        """
        self.catalog = None
        self.file_name = file_name
        if dataframe is not None:
            self.catalog = dataframe
            if self.file_name is not None:
                raise UnableToLoadMetadata

    def load(
        self,
        station: Station,
        compression="8bit",
        night_time=True,
        target_datetimes: Optional[list] = None,
    ) -> Generator[Metadata, None, None]:
        """Load the metadata from the catalog.

        :param station: The station which impact which target, target_1h,
            target_3h and target_6h that will be included in the metadata.
        :param compression: If the image_path point to a compressed image.
            Possible values are [None, 8bit, 16bit] (Default 8bit)
        :param night_time: If the night time must be included.
            The night time is calculated depending of the station.

        :return: A generator of metadata which drops all rows missing a picture.
        """
        if (
            self.catalog is None
        ):  # We do not want to reload the catalog each time we load data.
            catalog = self._load_file()
        else:
            catalog = self.catalog

        image_column = self._image_column(compression)
        image_offset_column = self._image_column(compression, variable="offset")
        catalog = self._filter_null(catalog, image_column)
        catalog = self._filter_null(catalog, f"{station.name}_GHI")
        catalog = self._filter_night(catalog, station, night_time)
        self.compression = compression
        i = 0

        if (
            target_datetimes is not None
        ):  # The dataloader will supply a list of target date times.
            # catalog = catalog[catalog.index.isin(target_datetimes)] # TODO: Refactor in a function, for consistency
            for target_datetime in target_datetimes:
                yield self._build_metadata(
                    catalog,
                    station,
                    image_column,
                    image_offset_column,
                    pd.Timestamp(target_datetime),
                    catalog[target_datetime].iloc[0],
                )
                i = i + 1
            return

        for index, row in catalog.iterrows():
            yield self._build_metadata(
                catalog, station, image_column, image_offset_column, index, row
            )

    def _find_future_target(
        self,
        catalog: pd.DataFrame,
        station: Station,
        time: pd.Timestamp,
        hour: int,
        variable: str = "GHI",
    ):  # Python allows us to have a single definition for a function regardless of
        # datatype instead of relying on templating or reimplementation of the same logic
        # for different data types. This is why I removed the type here. Hope you won't mind ;)
        index = time + pd.to_timedelta(hour, unit="h")
        try:
            return catalog.loc[index][f"{station.name}_" + variable]
        except KeyError:
            return None

    # Can be used for either path or offset.
    def _image_column(self, compression: str, variable="path"):

        if compression is None:
            if variable == "path":
                return "ncdf_path"
            else:
                return None

        elif compression == "8bit":
            return "hdf5_8bit_" + variable
        elif compression == "16bit":
            return "hdf5_16bit_" + variable
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
        image_offset_column: str,
        timestamp: pd.Timestamp,
        row: pd.Series,
    ) -> Metadata:
        image_path = row[image_column]
        if image_offset_column is not None:
            image_offset = row[image_offset_column]
        else:
            image_offset = 0  # No offset for ncdf files. We just output 0 everytime.
        latitude = self.Stations[station.name][0]
        longitude = self.Stations[station.name][1]
        altitude = self.Stations[station.name][2]

        target_ghi = row[f"{station.name}_GHI"]
        target_ghi_1h = self._find_future_target(
            catalog, station, timestamp, 1, variable="GHI"
        )
        target_ghi_3h = self._find_future_target(
            catalog, station, timestamp, 3, variable="GHI"
        )
        target_ghi_6h = self._find_future_target(
            catalog, station, timestamp, 6, variable="GHI"
        )

        target_cloudiness = row[f"{station.name}_CLOUDINESS"]
        target_cloudiness_1h = self._find_future_target(
            catalog, station, timestamp, 1, variable="CLOUDINESS"
        )
        target_cloudiness_3h = self._find_future_target(
            catalog, station, timestamp, 3, variable="CLOUDINESS"
        )
        target_cloudiness_6h = self._find_future_target(
            catalog, station, timestamp, 6, variable="CLOUDINESS"
        )

        datetime = timestamp.to_pydatetime()

        return Metadata(
            image_path=image_path,
            datetime=datetime,
            image_compression=self.compression,
            image_offset=image_offset,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            target_ghi=target_ghi,
            target_ghi_1h=target_ghi_1h,
            target_ghi_3h=target_ghi_3h,
            target_ghi_6h=target_ghi_6h,
            target_cloudiness=target_cloudiness,
            target_cloudiness_1h=target_cloudiness_1h,
            target_cloudiness_3h=target_cloudiness_3h,
            target_cloudiness_6h=target_cloudiness_6h,
        )

    def _load_file(self) -> pd.DataFrame:
        try:
            with open(self.file_name, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError as e:
            raise UnableToLoadMetadata(e)
