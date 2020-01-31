import unittest
from datetime import datetime
from typing import Optional
from unittest import mock

import numpy as np

from src.data.dataloader import (DataLoader, ImageReader, InvalidImageChannel,
                                 InvalidImageOffSet, InvalidImagePath)
from src.data.metadata import Coordinates, Metadata

ANY_COMPRESSION = "8bits"
ANY_IMAGE_OFFSET = 6
ANY_DATETIME = datetime.now()
ANY_COORDINATES = Coordinates(10, 10, 10)

CHANNEL_ID = "ch1"
INVALID_CHANNEL_ID = "ch5"

INVALID_IMAGE_PATH = "path/to/nothing"
IMAGE_PATH = "tests/data/samples/2015.11.01.0800.h5"
IMAGE = np.random.randint(low=0, high=255, size=(50, 50))


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.image_reader = mock.MagicMock(ImageReader)
        self.dataloader = DataLoader(self.image_reader)

    def test_givenOneMetadata_whenCreateDataset_shouldReadImage(self):
        self.image_reader.read = mock.Mock(return_value=IMAGE)

        dataset = self.dataloader.create_dataset(self._metadata_iterable(IMAGE_PATH))

        for image, target in dataset:
            self.assertTrue(np.array_equal(IMAGE, image.numpy()))

    def test_givenOneMetadata_whenCreateDataset_shouldReadOneImage(self):
        self.image_reader.read = mock.Mock(return_value=IMAGE)

        dataset = self.dataloader.create_dataset(self._metadata_iterable(IMAGE_PATH))

        self.assertEqual(1, num_elems(dataset))

    def test_givenOneMetadata_whenCreateDataset_shouldReturnTarget(self):
        self.image_reader.read = mock.Mock(return_value=IMAGE)
        targets = np.array([2, 3, 4, 5])
        dataset = self.dataloader.create_dataset(
            self._metadata_iterable(
                IMAGE_PATH,
                target_ghi=targets[0],
                target_ghi_1h=targets[1],
                target_ghi_3h=targets[2],
                target_ghi_6h=targets[3],
            )
        )

        for image, actual_targets in dataset:
            self.assertTrue(np.array_equal(actual_targets, targets))

    def _metadata_iterable(
        self,
        image_path: str,
        target_ghi: Optional[float] = None,
        target_ghi_1h: Optional[float] = None,
        target_ghi_3h: Optional[float] = None,
        target_ghi_6h: Optional[float] = None,
    ):
        yield Metadata(
            image_path,
            ANY_COMPRESSION,
            ANY_IMAGE_OFFSET,
            ANY_DATETIME,
            ANY_COORDINATES,
            target_ghi=target_ghi,
            target_ghi_1h=target_ghi_1h,
            target_ghi_3h=target_ghi_3h,
            target_ghi_6h=target_ghi_6h,
        )


class ImageReaderTest(unittest.TestCase):
    def setUp(self):
        self.image_reader = ImageReader()

    def test_whenReadImage_shouldReturnNumpyArray(self):
        image = self.image_reader.read(IMAGE_PATH, 0)

        self.assertEqual(type(image), np.ndarray)

    def test_givenInvalidOffSet_whenReadImage_shouldRaiseException(self):
        self.assertRaises(
            InvalidImageOffSet, lambda: self.image_reader.read(IMAGE_PATH, 100)
        )

    def test_given3Channels_whenReadImage_shouldReturn3Dimension(self):
        self.image_reader = ImageReader(channels=["ch1", "ch2", "ch3"])

        image = self.image_reader.read(IMAGE_PATH, 0)

        self.assertEqual(image.shape[2], 3)

    def test_givenInvalidChannel_whenReadImage_shouldRaiseException(self):
        self.image_reader = ImageReader(channels=[INVALID_CHANNEL_ID])

        self.assertRaises(
            InvalidImageChannel, lambda: self.image_reader.read(IMAGE_PATH, 0)
        )

    def test_givenInvalidPath_whenReadImage_shouldRaise(self):
        self.image_reader = ImageReader(channels=[CHANNEL_ID])

        self.assertRaises(
            InvalidImagePath, lambda: self.image_reader.read(INVALID_IMAGE_PATH, 0)
        )


def num_elems(iterable):
    return sum(1 for e in iterable)
