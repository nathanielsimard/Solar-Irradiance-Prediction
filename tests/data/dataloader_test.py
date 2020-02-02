import unittest
from datetime import datetime
from typing import Optional
from unittest import mock

import numpy as np

from src.data.dataloader import DataLoader
from src.data.image import ImageReader
from src.data.metadata import Coordinates, Metadata, MetadataLoader

import tests.data.metadata_test as metadata_test

ANY_COMPRESSION = "8bits"
ANY_IMAGE_OFFSET = 6
ANY_DATETIME = datetime.now()
ANY_COORDINATES = Coordinates(10, 10, 10)

FAKE_IMAGE = np.random.randint(low=0, high=255, size=(50, 50))

IMAGE_PATH = "tests/data/samples/2015.11.01.0800.h5"


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.image_reader = mock.MagicMock(ImageReader)
        self.dataloader = DataLoader(self.image_reader)

    def test_givenOneMetadata_whenCreateDataset_shouldReadImage(self):
        self.image_reader.read = mock.Mock(return_value=FAKE_IMAGE)

        dataset = self.dataloader.create_dataset(self._metadata_iterable(IMAGE_PATH))

        for image, target in dataset:
            self.assertTrue(np.array_equal(FAKE_IMAGE, image.numpy()))

    def test_givenOneMetadata_whenCreateDataset_shouldReadOneImage(self):
        self.image_reader.read = mock.Mock(return_value=FAKE_IMAGE)

        dataset = self.dataloader.create_dataset(self._metadata_iterable(IMAGE_PATH))

        self.assertEqual(1, num_elems(dataset))

    def test_givenOneMetadata_whenCreateDataset_shouldReturnTarget(self):
        self.image_reader.read = mock.Mock(return_value=FAKE_IMAGE)
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

    def test_transform_image_path_no_transform(self):
        self.dataloader.local_path = None
        new_path = self.dataloader._transform_image_path(
            "/project/cq-training-1/project1/data/hdf5v7_8bit/2010.01.11.0800.h5"
        )
        self.assertEqual(
            new_path,
            "/project/cq-training-1/project1/data/hdf5v7_8bit/2010.01.11.0800.h5",
        )

    def test_transform_image_path(self):
        local_path = "/home/raphael/MILA/ift6759/project1_data/hdf5v7_8bit/"
        self.dataloader.local_path = local_path
        new_path = self.dataloader._transform_image_path(
            "/project/cq-training-1/project1/data/hdf5v7_8bit/2010.01.11.0800.h5"
        )
        self.assertEqual(
            new_path,
            "/home/raphael/MILA/ift6759/project1_data/hdf5v7_8bit/2010.01.11.0800.h5",
        )

    def test_create_dataset_once(self):
        dl = DataLoader(self.image_reader)
        loader = MetadataLoader(metadata_test.CATALOG_PATH)
        metadata = loader.load(
            metadata_test.A_STATION,
            metadata_test.A_STATION_COORDINATE,
            compression=None,
        )
        dl.create_dataset(metadata)
        self.assertRaises(ValueError, callable=dl.create_dataset, args=metadata)

    def test_parse_config_local_path(self):
        dl = DataLoader(self.image_reader, config={"LOCAL_PATH": "test"})
        self.assertEqual(dl.local_path, "test")
        dl = DataLoader(self.image_reader)
        self.assertEqual(dl.local_path, None)

    def test_parse_config_skip_missing(self):
        dl = DataLoader(self.image_reader, config={"SKIP_MISSING": True})
        self.assertEqual(dl.skip_missing, True)
        dl = DataLoader(self.image_reader)
        self.assertEqual(dl.skip_missing, False)

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


def num_elems(iterable):
    return sum(1 for e in iterable)
