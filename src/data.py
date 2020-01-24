import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Metadata:
    image: str


class MetadataLoader:
    """Load the metadata from the catalog for differents stations."""

    def __init__(self, file_name: str) -> None:
        """Create a metadata loader.

        :param file_name: path to the catalog file.
        """
        self.file_name = file_name

    def load_all_stations(self, compression="8bits") -> List[Metadata]:
        catalog = self._load_file()
        catalog = catalog[catalog.hdf5_8bit_path != "nan"]

        stations_metadata = []
        for index, row in catalog.iterrows():
            image = self._find_image(row, compression)
            metadata = Metadata(image)
            stations_metadata.append(metadata)

        return stations_metadata

    def load_stations(self, stationid="BND", compression="8bits"):
        pass

    def _find_image(self, catalog_entry: pd.Series, compression: str) -> str:
        if compression == "16bits":
            return catalog_entry.hdf5_16bit_path
        else:
            return catalog_entry.hdf5_8bit_path

    def _load_file(self) -> pd.DataFrame:
        with open(self.file_name, "rb") as file:
            return pickle.load(file)
