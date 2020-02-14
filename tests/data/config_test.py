import unittest
from datetime import datetime

from src.data.config import read_configuration_file
from src.data.metadata import Station

DUMMY_TRAIN_CFG_PATH = "tests/data/samples/dummy_train_cfg.json"
DUMMY_TEST_CFG_PATH = "tests/data/samples/dummy_test_cfg.json"


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = read_configuration_file(DUMMY_TEST_CFG_PATH)

    def test_load_catalog_path(self):
        self.assertEqual(len(self.config.catalog), 35040)

    def test_load_target_datetimes(self):
        self.assertEqual(self.config.target_datetimes[10], datetime(2015, 1, 6, 16))

    def test_load_stations(self):
        self.assertAlmostEqual(self.config.stations[Station.BND].longitude, -88.37309)
        self.assertAlmostEqual(self.config.stations[Station.BND].latitude, 40.05192)
        self.assertAlmostEqual(self.config.stations[Station.BND].altitude, 230)

    def test_load_target_offsets(self):
        self.assertEqual(
            self.config.target_time_offsets,
            ["P0DT0H0M0S", "P0DT1H0M0S", "P0DT3H0M0S", "P0DT6H0M0S"],
        )
