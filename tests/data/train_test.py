import unittest

from src.data import dataloader, metadata, train
from src.data.metadata import Metadata, MetadataLoader

CATALOG_PATH = "tests/data/samples/catalog-test.pkl"

A_STATION = metadata.Station.BND
A_STATION_COORDINATE = metadata.Coordinates(40.05192, -88.37309, 230)


class TrainIntegrationTest(unittest.TestCase):
    def test_load_metadata(self):
        metadata_loader = MetadataLoader(file_name=CATALOG_PATH)
        timestamps = metadata_loader.catalog.index.tolist()
        datetimes = [timestamp.to_pydatetime() for timestamp in timestamps]

        metadata = train.metadata_station(metadata_loader, datetimes)

        for m in metadata:
            self.assertTrue(isinstance(m, Metadata))
            break

    def test_itarate_multiple_times(self):
        config = train.default_config()
        config.error_strategy = dataloader.ErrorStrategy.ignore
        config.features = [dataloader.Feature.target_ghi]
        metadata_loader = metadata.MetadataLoader(file_name=CATALOG_PATH)
        dataset = dataloader.create_dataset(
            metadata_loader.load(A_STATION, A_STATION_COORDINATE), config=config
        )

        first_run_data = 0
        for data in dataset:
            first_run_data += 1

        second_run_data = 0
        for data in dataset:
            second_run_data += 1

        self.assertEqual(first_run_data, second_run_data)
