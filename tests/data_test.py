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

    def test_load_metadata_image_offset_with_8bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, compression="8bit", night_time=False)
        actual = next(metadata).image_offset
        self.assertAlmostEqual(actual, 22)

    def test_load_metadata_image_offset_with_16bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, compression="16bit", night_time=False)
        actual = next(metadata).image_offset
        self.assertAlmostEqual(actual, 22)

    def test_load_metadata_image_offset_with_no_compression(self):
        loader = MetadataLoader(CATALOG_PATH)
        metadata = loader.load(A_STATION, compression=None, night_time=False)
        actual = next(metadata).image_offset
        self.assertAlmostEqual(actual, 0)

    def test_load_metadata_datatime(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION)

        expected_data = datetime(2010, 1, 1, 8, 0, 0, 0)
        self.assertEqual(expected_data, next(metadata).datetime)

    def test_load_metadata_target_ghi_(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target = -3.986666666666666

        metadata = loader.load(station_with_target)

        actual_target: Any = next(metadata).target_ghi
        self.assertAlmostEqual(target, actual_target)

    def test_load_metadata_target_ghi_1hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_1h = -3.926666666666665

        metadata = loader.load(station_with_target)

        actual_target_1h: Any = next(metadata).target_ghi_1h
        self.assertAlmostEqual(target_1h, actual_target_1h)

    def test_load_metadata_target_ghi_3hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_3h = -3.720000000000001

        metadata = loader.load(station_with_target)

        actual_target_3h: Any = next(metadata).target_ghi_3h
        self.assertAlmostEqual(target_3h, actual_target_3h)

    def test_load_metadata_target_ghi_6hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_6h = 29.10666666666667

        metadata = loader.load(station_with_target)

        actual_target_6h: Any = next(metadata).target_ghi_6h
        self.assertAlmostEqual(target_6h, actual_target_6h)

    def test_load_metadata_target_cloudiness(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target = "night"

        metadata = loader.load(station_with_target)

        actual_target: Any = next(metadata).target_cloudiness
        self.assertAlmostEqual(target, actual_target)

    def test_load_metadata_target_cloudiness_1hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_1h = "night"

        metadata = loader.load(station_with_target)

        actual_target_1h: Any = next(metadata).target_cloudiness_1h
        self.assertAlmostEqual(target_1h, actual_target_1h)

    def test_load_metadata_target_cloudiness_3hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_3h = "night"

        metadata = loader.load(station_with_target)

        actual_target_3h: Any = next(metadata).target_cloudiness_3h
        self.assertAlmostEqual(target_3h, actual_target_3h)

    def test_load_metadata_target_cloudiness_6hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_6h = "variable"

        metadata = loader.load(station_with_target)

        actual_target_6h: Any = next(metadata).target_cloudiness_6h
        self.assertAlmostEqual(target_6h, actual_target_6h)

    # "BND": (40.05192, -88.37309, 230)
    def test_load_metadata_latitude(self):
        loader = MetadataLoader(CATALOG_PATH)
        metadata = loader.load(Station.BND, night_time=False)
        actual_latitude: Any = next(metadata).latitude
        self.assertAlmostEqual(40.05192, actual_latitude)

    def test_load_metadata_longitude(self):
        loader = MetadataLoader(CATALOG_PATH)
        metadata = loader.load(Station.BND, night_time=False)
        actual_longitude: Any = next(metadata).longitude
        self.assertAlmostEqual(-88.37309, actual_longitude)

    def test_load_metadata_altitude(self):
        loader = MetadataLoader(CATALOG_PATH)
        metadata = loader.load(Station.BND, night_time=False)
        actual_altitude: Any = next(metadata).altitude
        self.assertAlmostEqual(230, actual_altitude)

    def test_load_metadata_with_night_time(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, night_time=True)

        num_metadata = self._num_metadata(metadata)
        self.assertEqual(NUM_METADATA, num_metadata)

    def test_load_metadata_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, night_time=True, compression="8bit")
        actual: Any = next(metadata).image_compression
        self.assertEqual(actual, "8bit")

        metadata = loader.load(A_STATION, night_time=True, compression="16bit")
        actual: Any = next(metadata).image_compression
        self.assertEqual(actual, "16bit")


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
