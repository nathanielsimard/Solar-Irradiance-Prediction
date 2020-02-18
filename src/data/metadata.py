import pickle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

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


@dataclass
class Coordinates:
    """Simple coordinates on earth."""

    latitude: float
    longitude: float
    altitude: float

    def __str__(self):
        """Convertes coordinates to string. Will be used for logging."""
        return f"({self.latitude}, {self.longitude}, {self.altitude})"


class UnableToLoadMetadata(Exception):
    """Exception that occured when something wrong happened when loading metadata."""

    pass


@dataclass
class Metadata:
    """Metadata containing the information necessary to train some models.

    Attributes:
        image_paths: Full path on helios
        image_compression: "8bit", "16bit" or "None"
        image_offsets: Indice de l'image dans le fichier hd5
        datetime: UTC datetime
        coordinates: Coordinates
        target_ghi: GHI, non normalized, watts/m2
        target_ghi_1h: Same, T+1h
        target_ghi_3h: Same, T+3h
        target_ghi_6h: Same, T+6h
        target_cloudiness: "night", #"cloudy", "clear", "variable" or "slightly cloudy"
        target_cloudiness_1h: Same, T+1h
        target_cloudiness_3h: Same, T+3h
        target_cloudiness_6h: Same, T+6h
    """

    image_paths: List[str]
    image_compression: str
    image_offsets: List[int]
    datetime: datetime
    coordinates: Coordinates
    target_ghi: Optional[float] = None
    target_ghi_1h: Optional[float] = None
    target_ghi_3h: Optional[float] = None
    target_ghi_6h: Optional[float] = None
    target_cloudiness: Optional[str] = None
    target_cloudiness_1h: Optional[str] = None
    target_cloudiness_3h: Optional[str] = None
    target_cloudiness_6h: Optional[str] = None
    target_clearsky: Optional[float] = None
    target_clearsky_1h: Optional[float] = None
    target_clearsky_3h: Optional[float] = None
    target_clearsky_6h: Optional[float] = None

    def __str__(self):
        """Converts metadata to string for logging. Not all info is output."""
        return f"{self.image_paths}, {self.image_offsets}, {self.datetime}, {self.coordinates}"


class MetadataLoader:
    """Load the metadata from the catalog for differents stations."""

    # Dictionnary of all SURFRAD station names and locations
    # Given as (latitude, longitude, altitude) tuples

    def __init__(self, file_name=None, dataframe=None) -> None:
        """Create a metadata loader.

        :param file_name: Path to the catalog file.
            The file is supposed to be a pickle file
            containing a pandas dataframe.

        :param dataframe: The pandas datafram as loader by the dataloader. Can be used instead
            of the path to the actual dataframe when required.

            Those parameters are mutually exclusive and should not be provided at the same time.

        """
        if (
            not isinstance(file_name, str) and file_name is not None
        ):  # Calling this with a bool (I made the mistake)
            # make the code hang, impossible to break. No Exception raised. Just a silent hang.
            raise ValueError("File name provided must be a string!")
        self.catalog = None
        self.file_name = file_name
        if dataframe is not None:
            self.catalog = dataframe
            if self.file_name is not None:
                raise ValueError(
                    "A filename and catalog should not be provided at the same time."
                )
        if (  # We do not want to reload the catalog each time we load data.
            self.catalog is None
        ):
            self.catalog = self._load_file()

    def load(
        self,
        station: Station,
        coordinates: Coordinates,
        compression="8bit",
        night_time=True,
        skip_missing=True,
        target_datetimes: Optional[List[datetime]] = None,
        num_images=1,
        time_interval_min=15,
    ) -> Iterable[Metadata]:
        """Load the metadata from the catalog.

        Args:
            station: The station which impact which target, target_1h,
                target_3h and target_6h that will be included in the metadata.
            coordinates: Coordinates of the given station.
            compression: If the image_path point to a compressed image.
                Possible values are [None, 8bit, 16bit] (Default 8bit)
            night_time: If the night time must be included.
                The night time is calculated depending of the station.
            skip_missing: If False, it will raise an exception when
                trying to generate metadata for an unknow datetime.
            target_datetimes: The target time to fetch metadata.
                If none is probided, all datetimes are considered.
            num_images: Number of images to return. If more than
                1 is probided, image from the past with time interval
                are going to be added to image_paths.
            time_interval_min: Time interval between images in minutes.

        :return: A generator of metadata which drops all rows missing a picture.
        """
        catalog = self.catalog

        if not isinstance(coordinates, Coordinates):
            raise ValueError("Must provide a Coordinate object")

        image_column = self._image_column(compression)
        image_offset_column = self._image_column(compression, variable="offset")

        catalog = self._filter_null(catalog, image_column)
        catalog = self._filter_null(catalog, f"{station.name}_GHI")
        catalog = self._filter_null(catalog, f"{station.name}_CLEARSKY_GHI")
        catalog = self._filter_night(catalog, station, night_time)

        target_timestamps = self._target_timestamps(catalog, target_datetimes)
        catalog = catalog.drop_duplicates()
        rows = catalog.to_dict("index")

        for i, target_timestamp in enumerate(target_timestamps):
            try:
                row = rows[target_timestamp]
            except KeyError as e:
                if skip_missing:
                    continue
                raise UnableToLoadMetadata(e)

            yield self._build_metadata(
                rows,
                station,
                coordinates,
                image_column,
                image_offset_column,
                target_timestamp,
                row,
                compression,
                num_images,
                time_interval_min,
            )

    def _find_future_value(
        self,
        rows: Dict[pd.Timestamp, Dict[str, Any]],
        station: Station,
        time: pd.Timestamp,
        hour: int,
        variable: str = "GHI",
    ):  # Python allows us to have a single definition for a function regardless of
        # datatype instead of relying on templating or reimplementation of the same logic
        # for different data types. This is why I removed the type here. Hope you won't mind ;)
        index = time + pd.to_timedelta(hour, unit="h")
        try:
            return rows[index][f"{station.name}_" + variable]
        except KeyError:
            return None

    # Can be used for either path or offset.
    def _image_column(self, compression: str, variable="path"):

        if compression is None:
            if variable == "path":
                return "ncdf_path"
            else:
                return

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
        rows: Dict[pd.Timestamp, Dict[str, Any]],
        station: Station,
        coordinates: Coordinates,
        image_column: str,
        image_offset_column: str,
        timestamp: pd.Timestamp,
        row: Dict[str, Any],
        compression: str,
        num_images: int,
        time_interval_min: int,
    ) -> Metadata:
        image_paths, image_offsets = self._find_additional_image_paths(
            rows,
            image_column,
            image_offset_column,
            timestamp,
            num_images,
            time_interval_min,
        )

        target_ghi = row[f"{station.name}_GHI"]
        target_ghi_1h = self._find_future_value(
            rows, station, timestamp, 1, variable="GHI"
        )
        target_ghi_3h = self._find_future_value(
            rows, station, timestamp, 3, variable="GHI"
        )
        target_ghi_6h = self._find_future_value(
            rows, station, timestamp, 6, variable="GHI"
        )

        target_cloudiness = row[f"{station.name}_CLOUDINESS"]
        target_cloudiness_1h = self._find_future_value(
            rows, station, timestamp, 1, variable="CLOUDINESS"
        )
        target_cloudiness_3h = self._find_future_value(
            rows, station, timestamp, 3, variable="CLOUDINESS"
        )
        target_cloudiness_6h = self._find_future_value(
            rows, station, timestamp, 6, variable="CLOUDINESS"
        )

        target_clearsky = row[f"{station.name}_CLEARSKY_GHI"]
        target_clearsky_1h = self._find_future_value(
            rows, station, timestamp, 1, variable="CLEARSKY_GHI"
        )
        target_clearsky_3h = self._find_future_value(
            rows, station, timestamp, 3, variable="CLEARSKY_GHI"
        )
        target_clearsky_6h = self._find_future_value(
            rows, station, timestamp, 6, variable="CLEARSKY_GHI"
        )

        datetime = timestamp.to_pydatetime()

        return Metadata(
            image_paths=image_paths,
            datetime=datetime,
            image_compression=compression,
            image_offsets=image_offsets,
            coordinates=coordinates,
            target_ghi=target_ghi,
            target_ghi_1h=target_ghi_1h,
            target_ghi_3h=target_ghi_3h,
            target_ghi_6h=target_ghi_6h,
            target_cloudiness=target_cloudiness,
            target_cloudiness_1h=target_cloudiness_1h,
            target_cloudiness_3h=target_cloudiness_3h,
            target_cloudiness_6h=target_cloudiness_6h,
            target_clearsky=target_clearsky,
            target_clearsky_1h=target_clearsky_1h,
            target_clearsky_3h=target_clearsky_3h,
            target_clearsky_6h=target_clearsky_6h,
        )

    def _find_additional_image_paths(
        self,
        rows,
        image_column,
        image_offset_column,
        timestamp,
        num_images,
        time_interval_min,
    ):
        image_paths = []
        image_offsets = []
        # Iterate in reverse to add the oldest images first.
        for i in range(num_images - 1, -1, -1):
            index = timestamp - pd.to_timedelta(i * time_interval_min, unit="min")
            try:
                image_paths.append(rows[index][image_column])
                if image_offset_column is not None:
                    image_offsets.append(rows[index][image_offset_column])
                else:
                    image_offsets.append(0)
            except KeyError:
                image_paths.append("/unknow/path")  # Will be handle by the dataloader
                image_offsets.append(0)

        return image_paths, image_offsets

    def _find_image_path(
        self, rows: Dict[pd.Timestamp, Any], num_images, time_interval_min
    ):
        pass

    def _target_timestamps(
        self, catalog: pd.DataFrame, target_datetimes: Optional[List[datetime]]
    ) -> List[datetime]:
        if target_datetimes is None:
            return catalog.index.tolist()

        return [pd.Timestamp(target_datetime) for target_datetime in target_datetimes]

    def _load_file(self) -> pd.DataFrame:
        try:
            with open(self.file_name, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError as e:
            raise UnableToLoadMetadata(e)
