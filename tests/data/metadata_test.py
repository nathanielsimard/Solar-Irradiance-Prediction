import pickle
import unittest
from datetime import datetime
from typing import Any, Generator

from src.data.metadata import MetadataLoader, Station, UnableToLoadMetadata

CATALOG_PATH = "tests/data/samples/catalog-test.pkl"

# Don't consider 'nan' or NaN values
NUM_METADATA = 2066
NUM_METADATA_BND_DAY_TIME = 1078


SOME_TARGET_DATETIMES = [
    datetime(2010, 6, 19, 22, 15),  # Only test timestamp that have images.
    datetime(2012, 3, 24, 12),
    datetime(2015, 9, 21, 21, 15),
    datetime(2012, 7, 6, 18),
    datetime(2014, 7, 13),
]

STATION_COORDINATES = {
    Station.BND: [40.05192, -88.37309, 230],
    Station.TBL: [40.12498, -105.23680, 1689],
    Station.DRA: [36.62373, -116.01947, 1007],
    Station.FPK: [48.30783, -105.10170, 634],
    Station.GWN: [34.25470, -89.87290, 98],
    Station.PSU: [40.72012, -77.93085, 376],
    Station.SXF: [43.73403, -96.62328, 473],
}

A_STATION = Station.BND
A_STATION_COORDINATE = STATION_COORDINATES[A_STATION]


class MetadataLoaderTest(unittest.TestCase):
    def test_load_metadata_with_bad_path(self):
        self.assertRaises(
            UnableToLoadMetadata, MetadataLoader, "path/that/doesnt/exist"
        )

    def test_load_metadata_image_path_without_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, A_STATION_COORDINATE, compression=None)

        first_image_path = next(metadata).image_path
        self.assertTrue("netcdf" in first_image_path)

    def test_load_metadata_image_path_with_8bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, A_STATION_COORDINATE, compression="8bit")

        first_image_path = next(metadata).image_path
        self.assertTrue("8bit" in first_image_path)

    def test_load_metadata_image_path_with_16bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, A_STATION_COORDINATE, compression="16bit")

        first_image_path = next(metadata).image_path
        self.assertTrue("16bit" in first_image_path)

    def test_load_metadata_image_offset_with_8bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(
            A_STATION, A_STATION_COORDINATE, compression="8bit", night_time=False
        )
        actual = next(metadata).image_offset
        self.assertAlmostEqual(actual, 22)
        self.assertIsInstance(actual, int)

    def test_load_metadata_image_offset_with_16bit_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(
            A_STATION, A_STATION_COORDINATE, compression="16bit", night_time=False
        )
        actual = next(metadata).image_offset
        self.assertAlmostEqual(actual, 22)

    def test_load_metadata_image_offset_with_no_compression(self):
        loader = MetadataLoader(CATALOG_PATH)
        metadata = loader.load(
            A_STATION, A_STATION_COORDINATE, compression=None, night_time=False
        )
        actual = next(metadata).image_offset
        self.assertAlmostEqual(actual, 0)

    def test_load_metadata_datatime(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, A_STATION_COORDINATE)

        expected_data = datetime(2010, 1, 1, 8, 0, 0, 0)
        self.assertEqual(expected_data, next(metadata).datetime)

    def test_load_metadata_target_ghi_(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target = -3.986666666666666

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target: Any = next(metadata).target_ghi
        self.assertAlmostEqual(target, actual_target)

    def test_load_metadata_target_ghi_1hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_1h = -3.926666666666665

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target_1h: Any = next(metadata).target_ghi_1h
        self.assertAlmostEqual(target_1h, actual_target_1h)

    def test_load_metadata_target_ghi_3hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_3h = -3.720000000000001

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target_3h: Any = next(metadata).target_ghi_3h
        self.assertAlmostEqual(target_3h, actual_target_3h)

    def test_load_metadata_target_ghi_6hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_6h = 29.10666666666667

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target_6h: Any = next(metadata).target_ghi_6h
        self.assertAlmostEqual(target_6h, actual_target_6h)

    def test_load_metadata_target_cloudiness(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target = "night"

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target: Any = next(metadata).target_cloudiness
        self.assertAlmostEqual(target, actual_target)

    def test_load_metadata_target_cloudiness_1hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_1h = "night"

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target_1h = next(metadata).target_cloudiness_1h
        self.assertEqual(target_1h, actual_target_1h)

    def test_load_metadata_target_cloudiness_3hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_3h = "night"

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target_3h = next(metadata).target_cloudiness_3h
        self.assertEqual(target_3h, actual_target_3h)

    def test_load_metadata_target_cloudiness_6hour(self):
        loader = MetadataLoader(CATALOG_PATH)
        station_with_target = Station.BND
        target_6h = "variable"

        metadata = loader.load(station_with_target, A_STATION_COORDINATE)

        actual_target_6h = next(metadata).target_cloudiness_6h
        self.assertEqual(target_6h, actual_target_6h)

    def test_load_metadata_coodinates(self):
        loader = MetadataLoader(CATALOG_PATH)
        station = Station.BND
        coordinates = STATION_COORDINATES[station]

        metadata = loader.load(station, coordinates, night_time=False)

        actual_coordinates = next(metadata).coordinates
        self.assertEqual(coordinates, actual_coordinates)

    def test_load_metadata_target_datetimes(self):
        loader = MetadataLoader(CATALOG_PATH)
        target_datetimes = [
            datetime(2010, 6, 19, 22, 15),  # Only test timestamp that have images.
            datetime(2012, 3, 24, 12),
            datetime(2015, 9, 21, 21, 15),
            datetime(2012, 7, 6, 18),
            datetime(2014, 7, 13),
            datetime(2010, 8, 31, 20, 45),
            datetime(2015, 4, 16, 12, 45),
            datetime(2013, 4, 17, 16),
            datetime(2012, 8, 15),
            datetime(2010, 11, 14, 19, 15),
            datetime(2014, 7, 21, 14, 30),
            datetime(2011, 11, 22, 17, 30),
            datetime(2010, 8, 15, 23),
            datetime(2010, 5, 11, 19),
            datetime(2013, 2, 15, 14, 15),
            datetime(2011, 2, 8, 17, 45),
        ]
        target_offsets = [
            57,
            16,
            53,
            40,
            64,
            51,
            19,
            32,
            64,
            45,
            26,
            38,
            60,
            44,
            25,
            39,
        ]

        metadata = loader.load(
            A_STATION,
            A_STATION_COORDINATE,
            night_time=True,
            target_datetimes=target_datetimes,
        )
        i = 0
        for datapoint in metadata:
            self.assertIsInstance(datapoint.image_offset, int)
            self.assertEqual(datapoint.image_offset, target_offsets[i])
            i = i + 1
        self.assertEqual(len(target_datetimes), i)

    def test_load_metadata_with_night_time(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(A_STATION, A_STATION_COORDINATE, night_time=True)

        num_metadata = self._num_metadata(metadata)
        self.assertEqual(NUM_METADATA, num_metadata)

    def test_load_metadata_compression(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(
            A_STATION, A_STATION_COORDINATE, night_time=True, compression="8bit"
        )
        actual: Any = next(metadata).image_compression
        self.assertEqual(actual, "8bit")

        metadata = loader.load(
            A_STATION, A_STATION_COORDINATE, night_time=True, compression="16bit"
        )
        actual: Any = next(metadata).image_compression
        self.assertEqual(actual, "16bit")

    def test_load_metadata_without_night_time(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(Station.BND, A_STATION_COORDINATE, night_time=False)

        num_metadata = self._num_metadata(metadata)
        self.assertEqual(NUM_METADATA_BND_DAY_TIME, num_metadata)

    def _num_metadata(self, metadata: Generator) -> int:
        num = 0
        for m in metadata:
            num += 1
        return num

    def test_load_metadata_with_specified_dataframe(self):
        dummy_catalog = pickle.load(open("tests/data/samples/catalog-test.pkl", "rb"))
        loader = MetadataLoader(file_name=None, dataframe=dummy_catalog)
        station_with_target = Station.BND
        target_6h = 29.10666666666667
        metadata = loader.load(station_with_target, A_STATION_COORDINATE)
        actual_target_6h: Any = next(metadata).target_ghi_6h
        self.assertAlmostEqual(target_6h, actual_target_6h)

    def testGivenTargetDatetimes_whenLoad_shouldLoadMetadataInOrder(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(
            Station.BND, A_STATION_COORDINATE, target_datetimes=SOME_TARGET_DATETIMES
        )

        for md, expected_datetime in zip(metadata, SOME_TARGET_DATETIMES):
            self.assertEqual(md.datetime, expected_datetime)

    def testGivenTargetDatetimes_whenLoad_shouldLoadSameAmountOfMetadata(self):
        loader = MetadataLoader(CATALOG_PATH)

        metadata = loader.load(
            Station.BND, A_STATION_COORDINATE, target_datetimes=SOME_TARGET_DATETIMES
        )

        self.assertEqual(self._num_metadata(metadata), len(SOME_TARGET_DATETIMES))

    def _next_target(self, metadata: Generator):
        return next(metadata).target
