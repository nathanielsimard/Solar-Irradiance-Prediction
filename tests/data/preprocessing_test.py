import unittest

from src.data import preprocessing
from tests.helpers.metadata import IMAGE_PATH_1, create_dataset

BATCH_SIZE = 2


class PreprocessingIntegrationTest(unittest.TestCase):
    def test_givenBDNStation_whenCenterStation_Coordinates(self):
        channels = ["ch1", "ch2", "ch3"]
        dataset = create_dataset(channels)

        transformed_dataset = preprocessing.center_station_coordinates(dataset)

        channels_num = len(channels)
        for croped in transformed_dataset.batch(BATCH_SIZE):
            self.assertEqual((BATCH_SIZE, 64, 64, channels_num), croped.shape)

    def test_givenTooLargeOutputSize_whenCenterStation_shouldRaise(self):
        channels = ["ch1", "ch2", "ch3"]
        dataset = create_dataset(channels)

        self.assertRaises(
            ValueError,
            lambda: preprocessing.center_station_coordinates(
                dataset, output_size=(700, 64),
            ),
        )
