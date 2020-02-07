import unittest

from src.data import dataloader, metadata
from src.data.preprocessing import MinMaxScaling, find_target_ghi_minmax_value
from src.data.train import default_config

CATALOG_PATH = "tests/data/samples/catalog-test.pkl"

A_STATION = metadata.Station.BND
A_STATION_COORDINATE = metadata.Coordinates(40.05192, -88.37309, 230)


class MinMaxScalingTest(unittest.TestCase):
    def setUp(self):
        self.scaling = MinMaxScaling(-200, 200)

    def test_whenNormalize_shouldKeepProportion(self):
        value1 = self.scaling.normalize(-30)
        value2 = self.scaling.normalize(250)

        self.assertTrue(value2 > value1)

    def test_whenNormalize_shouldRetuenValueBetween01(self):
        value1 = self.scaling.normalize(-3)
        value2 = self.scaling.normalize(199)

        self.assertTrue(1 > value1)
        self.assertTrue(0 < value1)

        self.assertTrue(1 > value2)
        self.assertTrue(0 < value2)

    def test_givenPositiveNumber_whenNormalize_shouldBeAbleToRescaleToOriginal(self):
        original = 150
        scaled = self.scaling.normalize(original)

        actual_original = self.scaling.original(scaled)

        self.assertAlmostEqual(original, actual_original)

    def test_givenNegativeNumber_whenNormalize_shouldBeAbleToRescaleToOriginal(self):
        original = -45
        scaled = self.scaling.normalize(original)

        actual_original = self.scaling.original(scaled)

        self.assertAlmostEqual(original, actual_original)


class TargetGHIIntegrationTest(unittest.TestCase):
    def setUp(self):
        config = default_config()
        config.error_strategy = dataloader.ErrorStrategy.ignore
        config.features = [dataloader.Feature.target_ghi]

        metadata_loader = metadata.MetadataLoader(file_name=CATALOG_PATH)
        self.dataset = dataloader.create_dataset(
            lambda: metadata_loader.load(A_STATION, A_STATION_COORDINATE), config=config
        )

    def test_find_minmax_value(self):
        max_target, min_target = find_target_ghi_minmax_value(dataset=self.dataset)

        self.assertTrue(max_target > 0)
        self.assertTrue(min_target < 0)
