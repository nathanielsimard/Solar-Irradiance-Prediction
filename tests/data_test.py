import unittest
from datetime import datetime
from typing import Any, Generator

from src.data import MetadataLoader, Station, UnableToLoadMetadata

CATALOG_PATH = "tests/data/catalog-test.pkl"

# Don't consider 'nan' or NaN values
NUM_METADATA = 2066
NUM_METADATA_BND_DAY_TIME = 1078

A_STATION = Station.BND


class MetadataLoaderTest(unittest.TestCase):
    def test_load_metadata_with_bad_path(self):
        loader = MetadataLoader("path/that/doesnt/exist")
        metadata = loader.load(A_STATION)

        self.assertRaises(
            UnableToLoadMetadata, lambda: next(metadata),
        )

    def test_load_metadata_image_path_without_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, compression=None)

        first_image_path = next(metadata).image_path
        self.assertTrue("netcdf" in first_image_path)

    def test_load_metadata_image_path_with_8bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, compression="8bit")

        first_image_path = next(metadata).image_path
        self.assertTrue("8bit" in first_image_path)

    def test_load_metadata_image_path_with_16bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, compression="16bit")

        first_image_path = next(metadata).image_path
        self.assertTrue("16bit" in first_image_path)

    def test_load_metadata_datatime(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION)

        expected_data = datetime(2010, 1, 1, 8, 0, 0, 0)
        self.assertEqual(expected_data, next(metadata).datetime)

    def test_load_metadata_target(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target = -3.986666666666666

        metadata = loader.load(station_with_target)

        actual_target: Any = next(metadata).target
        self.assertAlmostEqual(target, actual_target)

    def test_load_metadata_target_1hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_1h = -3.926666666666665

        metadata = loader.load(station_with_target)

        actual_target_1h: Any = next(metadata).target_1h
        self.assertAlmostEqual(target_1h, actual_target_1h)

    def test_load_metadata_target_3hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_3h = -3.720000000000001

        metadata = loader.load(station_with_target)

        actual_target_3h: Any = next(metadata).target_3h
        self.assertAlmostEqual(target_3h, actual_target_3h)

    def test_load_metadata_target_6hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_6h = 29.10666666666667

        metadata = loader.load(station_with_target)

        actual_target_6h: Any = next(metadata).target_6h
        self.assertAlmostEqual(target_6h, actual_target_6h)

    def test_load_metadata_with_night_time(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, night_time=True)

        num_metadata = self._num_metadata(metadata)
        self.assertEqual(NUM_METADATA, num_metadata)

    def test_load_metadata_without_night_time(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(Station.BND, night_time=False)

        num_metadata = self._num_metadata(metadata)
        self.assertEqual(NUM_METADATA_BND_DAY_TIME, num_metadata)

    def _num_metadata(self, metadata: Generator) -> int:
        num = 0
        for m in metadata:
            num += 1
        return num

    def _next_target(self, metadata: Generator):
        return next(metadata).target
