import unittest

from src import train
from src.data.metadata import Metadata, MetadataLoader

CATALOG_PATH = "tests/data/samples/catalog-test.pkl"


class TrainIntegrationTest(unittest.TestCase):
    def test_load_metadata(self):
        metadata_loader = MetadataLoader(file_name=CATALOG_PATH)
        timestamps = metadata_loader.catalog.index.tolist()
        datetimes = [timestamp.to_pydatetime() for timestamp in timestamps]

        metadata = train.metadata_station(metadata_loader, datetimes)

        for m in metadata:
            self.assertTrue(isinstance(m, Metadata))
            break
