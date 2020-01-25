import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Metadata:
    image: str
    night_time: bool


class MetadataLoader:
    """Load the metadata from the catalog for differents stations."""

    def __init__(self, file_name: str) -> None:
        """Create a metadata loader.

        :param file_name: path to the catalog file.
        """
        self.file_name = file_name

    def load_stations(
        self, compression="8bit", station="all", night_time=False
    ) -> List[Metadata]:
        catalog = self._load_file()
        image_column = self._image_column(compression)

        catalog = catalog[catalog[image_column] != "nan"]
        catalog = catalog[catalog[image_column].notnull()]
        if night_time:
            catalog = catalog[catalog.BND_DAYTIME == 1]

        stations_metadata = []
        for index, row in catalog.iterrows():
            image = row[image_column]
            metadata = Metadata(image, row.BND_DAYTIME == 1)
            stations_metadata.append(metadata)

        return stations_metadata

    def _image_column(self, compression: str):
        if compression == "8bit":
            return "hdf5_8bit_path"
        elif compression == "16bit":
            return "hdf5_16bit_path"

    def _load_file(self) -> pd.DataFrame:
        with open(self.file_name, "rb") as file:
            return pickle.load(file)
