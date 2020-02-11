import unittest
from datetime import datetime
from typing import Optional
from unittest import mock

import numpy as np

import src.data.config as cf
import tests.data.config_test as config_test
from src.data.dataloader import (
    Config,
    DataLoader,
    ErrorStrategy,
    Feature,
    MetadataFeatureIndex,
    MissingTargetException,
    UnregognizedErrorStrategy,
    UnregognizedFeature,
    CorruptedImage,
    parse_config,
)
from src.data.image import ImageReader
from src.data.metadata import Coordinates, Metadata

ANY_COMPRESSION = "8bits"
ANY_IMAGE_OFFSET = 6
ANY_DATETIME = datetime.now()
ANY_COORDINATES = Coordinates(10, 10, 10)

FAKE_IMAGE = np.random.randint(low=0, high=255, size=(50, 50))

IMAGE_NAME = "2015.11.01.0800.h5"
IMAGE_PATH = f"tests/data/samples/{IMAGE_NAME}"

AN_EXCEPTION = UnregognizedErrorStrategy("Test exception")
AN_EXCEPTION_TYPE = UnregognizedErrorStrategy

AN_HANDLED_EXCEPTION = CorruptedImage("Fake corrupted image")


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.image_reader = mock.MagicMock(ImageReader)
        self.dataloader = DataLoader(lambda: [self._metadata()], self.image_reader)

    def test_givenOneMetadata_whenCreateDataset_shouldReadImage(self):
        self.image_reader.read = mock.Mock(return_value=FAKE_IMAGE)

        dataset = self.dataloader.generator()

        for image, target in dataset:
            self.assertTrue(np.array_equal(FAKE_IMAGE, image.numpy()))

    def test_givenOneMetadata_whenCreateDataset_shouldReadOneImage(self):
        self.image_reader.read = mock.Mock(return_value=FAKE_IMAGE)

        dataset = self.dataloader.generator()

        self.assertEqual(1, num_elems(dataset))

    def test_givenOneMetadata_whenCreateDataset_shouldReturnTarget(self):
        self.image_reader.read = mock.Mock(return_value=FAKE_IMAGE)
        targets = np.array([2, 3, 4, 5])
        self.dataloader = DataLoader(
            lambda: [
                self._metadata(
                    target_ghi=targets[0],
                    target_ghi_1h=targets[1],
                    target_ghi_3h=targets[2],
                    target_ghi_6h=targets[3],
                )
            ],
            self.image_reader,
        )

        for image, actual_targets in self.dataloader.generator():
            self.assertTrue(np.array_equal(actual_targets, targets))

    def test_givenNoLocalPath_shouldUseOriginalPath(self):
        self.dataloader = DataLoader(
            lambda: [self._metadata()], self.image_reader, Config(local_path=None),
        )

        dataset = self.dataloader.generator()
        list(dataset)  # Force evaluate the dataset

        self.image_reader.read.assert_called_with(
            IMAGE_PATH, mock.ANY, mock.ANY, mock.ANY
        )

    def test_givenLocalPath_shouldUseLocalPathAsRoot(self):
        local_path = "local/path/"
        self.dataloader = DataLoader(
            lambda: [self._metadata()],
            self.image_reader,
            Config(local_path=local_path),
        )

        dataset = self.dataloader.generator()
        list(dataset)

        expected_path = f"{local_path}{IMAGE_NAME}"
        self.image_reader.read.assert_called_with(
            expected_path, mock.ANY, mock.ANY, mock.ANY
        )

    def test_givenSkipErrorStrategy_whenImageLoaderException_shouldSkipToNextItem(
        self,
    ):
        self.image_reader.read = mock.Mock(side_effect=[AN_EXCEPTION, FAKE_IMAGE])

        self.dataloader = DataLoader(
            lambda: [self._metadata(), self._metadata()],
            self.image_reader,
            Config(error_strategy=ErrorStrategy.skip),
        )
        items = list(self.dataloader.generator())

        self.assertEqual(1, len(items))

    def test_givenStopErrorStrategy_whenImageLoaderException_shouldRaise(self):
        self.image_reader.read = mock.Mock(side_effect=[AN_EXCEPTION, FAKE_IMAGE])

        self.dataloader = DataLoader(
            lambda: [self._metadata(), self._metadata()],
            self.image_reader,
            Config(error_strategy=ErrorStrategy.stop),
        )
        self.assertRaises(AN_EXCEPTION_TYPE, lambda: list(self.dataloader.generator()))

    def test_givenIgnoreErrorStrategy_whenImageLoaderException_shouldReturnDummyImage(
        self,
    ):
        self.image_reader.read = mock.Mock(
            side_effect=[AN_HANDLED_EXCEPTION, FAKE_IMAGE]
        )
        channels = ["ch1", "ch2"]
        crop_size = [40, 40]
        num_channels = len(channels)
        expected_image_shape = crop_size + [num_channels]

        self.dataloader = DataLoader(
            lambda: [self._metadata(), self._metadata()],
            self.image_reader,
            Config(
                error_strategy=ErrorStrategy.ignore,
                crop_size=crop_size,
                channels=channels,
            ),
        )

        items = list(self.dataloader.generator())

        first_image = items[0][0]
        self.assertEqual(2, len(items))
        self.assertEqual(first_image.shape, expected_image_shape)

    def test_givenSkipErrorStrategy_whenMissingTarget_shouldSkipToNextItem(self,):
        self.dataloader = DataLoader(
            lambda: [self._metadata(target_ghi_1h=None), self._metadata()],
            self.image_reader,
            Config(error_strategy=ErrorStrategy.skip),
        )
        items = list(self.dataloader.generator())

        self.assertEqual(1, len(items))

    def test_givenStopErrorStrategy_whenMissingTarget_shouldRaise(self):
        self.dataloader = DataLoader(
            lambda: [self._metadata(), self._metadata(target_ghi_3h=None)],
            self.image_reader,
            Config(error_strategy=ErrorStrategy.stop),
        )
        self.assertRaises(
            MissingTargetException, lambda: list(self.dataloader.generator())
        )

    def test_givenIgnoreErrorStrategy_whenMissingTarget_shouldReturnDummyTarget(self,):
        self.dataloader = DataLoader(
            lambda: [self._metadata(target_ghi=None)],
            self.image_reader,
            Config(error_strategy=ErrorStrategy.ignore,),
        )

        for image, targets in self.dataloader.generator():
            for target in targets:
                self.assertIsNotNone(target)

    def test_parse_config_local_path(self):
        config = parse_config({"LOCAL_PATH": "test"})

        self.assertEqual(config.local_path, "test")

    def test_parse_config_crop_size(self):
        config = parse_config({"CROP_SIZE": (128, 128)})

        self.assertEqual(config.crop_size, (128, 128))

    def test_parse_config_features(self):
        config = parse_config({"FEATURES": ["image", "target_ghi", "metadata"]})

        self.assertEqual(
            config.features, [Feature.image, Feature.target_ghi, Feature.metadata],
        )

    def test_parse_error_stategy(self):
        config = parse_config({"ERROR_STATEGY": "skip"})

        self.assertEqual(
            config.error_strategy, ErrorStrategy.skip,
        )

    def test_parse_wrong_error_stategy(self):
        self.assertRaises(
            UnregognizedErrorStrategy,
            lambda: parse_config({"ERROR_STATEGY": ["wrong_format"]}),
        )

    def test_givenWrongFeature_whenParse_shouldRaise(self):
        self.assertRaises(
            UnregognizedFeature, lambda: parse_config({"FEATURES": ["wrong_format"]}),
        )

    def test_metadata_format(self):
        config = cf.read_configuration_file(config_test.DUMMY_TEST_CFG_PATH)
        metadata = Metadata(
            "",
            "",
            0,
            datetime=datetime(2010, 6, 19, 22, 15),
            coordinates=config.stations[cf.Station.BND],
        )

        self.dataloader = DataLoader(
            lambda: [metadata], self.image_reader, Config(features=[Feature.metadata]),
        )

        for (meta,) in self.dataloader.generator():
            self.assertCloseTo(meta[MetadataFeatureIndex.GHI_T], 471.675670)
            self.assertCloseTo(meta[MetadataFeatureIndex.GHI_T_1h], 280.165857)
            self.assertCloseTo(meta[MetadataFeatureIndex.GHI_T_3h], 0.397029)
            self.assertCloseTo(meta[MetadataFeatureIndex.GHI_T_6h], 0.0)
            self.assertCloseTo(meta[MetadataFeatureIndex.SOLAR_TIME], 0.0)

    def test_givenFeatures_whenCreateDataset_shouldReturnSameNumberOfFeatures(self):
        features = [
            Feature.image,
            Feature.target_ghi,
            Feature.metadata,
        ]

        self.dataloader = DataLoader(
            lambda: [self._metadata()], self.image_reader, Config(features=features),
        )

        for data in self.dataloader.generator():
            self.assertEqual(len(data), len(features))

    def assertCloseTo(self, value: float, target: float, epsilon: float = 0.001):
        self.assertAlmostEqual(value, target, delta=epsilon)

    def _metadata(
        self,
        image_path: str = IMAGE_PATH,
        target_ghi: Optional[float] = 100,
        target_ghi_1h: Optional[float] = 100,
        target_ghi_3h: Optional[float] = 100,
        target_ghi_6h: Optional[float] = 100,
    ):
        return Metadata(
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


def num_elems(iterable):
    return sum(1 for e in iterable)
