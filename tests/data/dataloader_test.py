import unittest
from datetime import datetime
from unittest import mock

import numpy as np

from src.data.dataloader import DataLoader, ImageReader
from src.data.metadata import Coordinates, Metadata

ANY_COMPRESSION = "8bits"
ANY_IMAGE_OFFSET = 6
ANY_DATETIME = datetime.now()
ANY_COORDINATES = Coordinates(10, 10, 10)

IMAGE_PATH = "image/path"
IMAGE = np.random.randint(low=0, high=255, size=(50, 50))


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.image_reader = mock.MagicMock(ImageReader)
        self.dataloader = DataLoader(self.image_reader)

    def test_givenOneMetadata_whenCreateDataset_shouldReadImage(self):
        self.image_reader.read = mock.Mock(return_value=IMAGE)

        dataset = self.dataloader.create_dataset(self._metadata_iterable(IMAGE_PATH))

        for first_element in dataset:
            self.assertTrue(np.array_equal(IMAGE, first_element.numpy()))

    def test_givenOneMetadata_whenCreateDataset_shouldReadOneImage(self):
        self.image_reader.read = mock.Mock(return_value=IMAGE)

        dataset = self.dataloader.create_dataset(self._metadata_iterable(IMAGE_PATH))

        self.assertEqual(1, num_elems(dataset))

    def _metadata_iterable(self, image_path: str):
        yield Metadata(
            image_path, ANY_COMPRESSION, ANY_IMAGE_OFFSET, ANY_DATETIME, ANY_COORDINATES
        )


def num_elems(iterable):
    return sum(1 for e in iterable)
