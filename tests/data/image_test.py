import unittest

import numpy as np

from src.data.image import (CorruptedImage, ImageReader, InvalidImageChannel,
                            InvalidImageOffSet, InvalidImagePath)
from src.data.metadata import Coordinates

ANY_COORDINATES = Coordinates(10, 10, 10)

CHANNEL_ID = "ch1"
INVALID_CHANNEL_ID = "ch5"

OFFSET = 0
MISSING_IMAGE_OFFSET = 4
INVALID_OFFSET = 100


INVALID_IMAGE_PATH = "path/to/nothing"
IMAGE_PATH = "tests/data/samples/2015.11.01.0800.h5"
COORDINATES = Coordinates(40.05192, -88.37309, 230)


class ImageReaderTest(unittest.TestCase):
    def setUp(self):
        self.image_reader = ImageReader()

    def test_whenReadImage_shouldReturnNumpyArray(self):
        image = self.image_reader.read(IMAGE_PATH, OFFSET, ANY_COORDINATES)

        self.assertEqual(type(image), np.ndarray)

    def test_givenInvalidOffSet_whenReadImage_shouldRaiseException(self):
        self.assertRaises(
            InvalidImageOffSet,
            lambda: self.image_reader.read(IMAGE_PATH, INVALID_OFFSET, ANY_COORDINATES),
        )

    def test_given3Channels_whenReadImage_shouldReturn3Dimension(self):
        self.image_reader = ImageReader(channels=["ch1", "ch2", "ch3"])

        image = self.image_reader.read(IMAGE_PATH, OFFSET, ANY_COORDINATES)

        self.assertEqual(image.shape[2], 3)

    def test_givenInvalidChannel_whenReadImage_shouldRaiseException(self):
        self.image_reader = ImageReader(channels=[INVALID_CHANNEL_ID])

        self.assertRaises(
            InvalidImageChannel,
            lambda: self.image_reader.read(IMAGE_PATH, OFFSET, ANY_COORDINATES),
        )

    def test_givenInvalidPath_whenReadImage_shouldRaise(self):
        self.image_reader = ImageReader(channels=[CHANNEL_ID])

        self.assertRaises(
            InvalidImagePath,
            lambda: self.image_reader.read(INVALID_IMAGE_PATH, OFFSET, ANY_COORDINATES),
        )

    def test_givenOutputSize_whenReadImageWithCoordinates_shouldReturnCropedImage(self):
        output_size = (64, 64)
        self.image_reader = ImageReader(output_size=output_size)

        image = self.image_reader.read(IMAGE_PATH, OFFSET, COORDINATES)

        image_shape_without_channel = image.shape[:2]
        self.assertEqual(output_size, image_shape_without_channel)

    def test_givenNoImage_whenRead_shouldRaise(self):
        self.assertRaises(
            CorruptedImage,
            lambda: self.image_reader.read(
                IMAGE_PATH, MISSING_IMAGE_OFFSET, ANY_COORDINATES
            ),
        )

    def test_givenOutOfBoudOutputSize_whenRead_shouldPadtheImageToFitTheOutputSize(
        self,
    ):
        output_size = (2000, 2000)
        self.image_reader = ImageReader(output_size=output_size)

        image = self.image_reader.read(IMAGE_PATH, OFFSET, COORDINATES)

        image_shape_without_channel = image.shape[:2]
        self.assertEqual(output_size, image_shape_without_channel)
