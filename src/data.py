import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Optional

import pandas as pd


class Station(Enum):
    BND = "BND"
    TBL = "TBL"
    DRA = "DRA"
    FPK = "FPK"
    GWN = "GWN"
    PSU = "PSU"
    SXF = "SXF"


class UnableToLoadMetadata(Exception):
    """Exception that occured when something wrong happened when loading stations."""

    pass


@dataclass
class Metadata:
    image_path: str
    target: Optional[float]
    target_1h: Optional[float]
    target_3h: Optional[float]
    target_6h: Optional[float]


class MetadataLoader:
    """Load the metadata from the catalog for differents stations."""

    def __init__(self, file_name: str) -> None:
        """Create a metadata loader.

        :param file_name: path to the catalog file.
        """
        self.file_name = file_name

    def load(
        self, station: Station, compression="8bit", night_time=True
    ) -> Generator[Metadata, None, None]:
        catalog = self._load_file()
        image_column = self._image_column(compression)

        catalog = catalog[catalog[image_column] != "nan"]
        catalog = catalog[catalog[image_column].notnull()]

        catalog = catalog[catalog[f"{station.name}_GHI"] != "nan"]
        catalog = catalog[catalog[f"{station.name}_GHI"].notnull()]

        if not night_time:
            catalog = catalog[catalog[f"{station.name}_DAYTIME"] == 1]

        for index, row in catalog.iterrows():
            image_path = row[image_column]
            target = row[f"{station.name}_GHI"]

            target_1h = self._find_futur_target(catalog, station, index, 1)
            target_3h = self._find_futur_target(catalog, station, index, 3)
            target_6h = self._find_futur_target(catalog, station, index, 6)

            yield Metadata(image_path, target, target_1h, target_3h, target_6h)

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

    def _load_file(self) -> pd.DataFrame:
        try:
            with open(self.file_name, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError as e:
            raise UnableToLoadMetadata(e)
