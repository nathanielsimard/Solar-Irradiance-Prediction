import unittest

from src.data import preprocessing
from tests.helpers.metadata import create_dataset

BATCH_SIZE = 2
CHANNELS = ["ch1", "ch2", "ch3"]
NUM_CHANNELS = len(CHANNELS)


class PreprocessingIntegrationTest(unittest.TestCase):
    def test_givenBDNStation_whenCenterStation_Coordinates(self):
        dataset = create_dataset(CHANNELS)

        transformed_dataset = preprocessing.center_station_coordinates(dataset)

        for croped in transformed_dataset.batch(BATCH_SIZE):
            self.assertEqual((BATCH_SIZE, 64, 64, NUM_CHANNELS), croped.shape)

    @unittest.skip("Flaky test, need to make sure the offsets are withing the image")
    def test_givenTooLargeOutputSize_whenCenterStation_shouldRaise(self):
        dataset = create_dataset(CHANNELS)

        transformed_dataset = preprocessing.center_station_coordinates(
            dataset, output_size=(800, 64)
        )

        self.assertRaises(ValueError, lambda: [i for i in transformed_dataset])
