from typing import Any, Dict, Iterable

import numpy as np
import tensorflow as tf
import h5py

from src.data.utils import fetch_hdf5_sample, viz_hdf5_imagery
from src.data import metadata


class InvalidImageOffSet(Exception):
    """Exception raised when offset isn't valid."""

    pass


class InvalidImageChannel(Exception):
    """Exception raised when channel isn't valid (valid channel: ch1, ch2, ch3, ch4, ch6)."""

    pass


class InvalidImagePath(Exception):
    """Exception when the path is not found."""

    pass


class ImageReader(object):
    """Read the images. Compression format is handle automaticly."""

    def __init__(self, channels=["ch1"]):
        """Default channel for image reading is ch1."""
        self.channels = channels

    def read(self, image_path: str, image_offset: int) -> np.ndarray:
        """Read image and return multidimensionnal numpy array."""
        try:
            file_reader = h5py.File(image_path)
        except OSError as e:
            raise InvalidImagePath(e)

        return np.stack(self._read_images(image_offset, file_reader))

    def _read_images(self, image_offset, file_reader):
        """Raise errors when invalid offset or channel while reading images."""
        try:
            return [
                fetch_hdf5_sample(channel, file_reader, image_offset)
                for channel in self.channels
            ]

        except ValueError as e:
            raise InvalidImageOffSet(e)
        except KeyError as e:
            raise InvalidImageChannel(e)

    def visualize(self, image_path: str, channel="ch1"):
        """Open amazing image window."""
        viz_hdf5_imagery(image_path, [channel])


class DataLoader(object):
    """Load the data from disk using tensorflow Dataset.

    To load a batch of data, you can iterate over the tf.data.Dataset by batch.
    >>>dataset=dataset.batch(batch_size)
    """

    def __init__(self, image_reader: ImageReader, config: Dict[str, Any] = {}) -> None:
        """Create a DataLoader with some user config."""
        self.image_reader = image_reader
        self.config = config

    def create_dataset(self, metadata: Iterable[metadata.Metadata]) -> tf.data.Dataset:
        """Create a tensorflow Dataset base on the metadata and dataloader's config.

        Targets are optional in Metadata. If one is missing, set it to zero.
        """

        def gen():
            for md in metadata:
                image = self.image_reader.read(md.image_path, md.image_offset)
                data = tf.convert_to_tensor(image, dtype=tf.int64)
                target = tf.constant(
                    [
                        _target_value(md.target_ghi),
                        _target_value(md.target_ghi_1h),
                        _target_value(md.target_ghi_3h),
                        _target_value(md.target_ghi_6h),
                    ]
                )
                yield (data, target)

        return tf.data.Dataset.from_generator(gen, (tf.int64, tf.float64))


def _target_value(target):
    if target is None:
        return 0
    return target
