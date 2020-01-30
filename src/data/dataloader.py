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


class ImageReader(object):
    """Read the images. Compression format is handle automaticly."""

    def __init__(self, channels=["ch1"]):
        """Default channel for image reading is ch1."""
        self.channels = channels

    def read(self, image_path: str, image_offset: int) -> np.ndarray:
        """Read image and return multidimensionnal numpy array."""
        file_reader = h5py.File(image_path)

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

    def visualize(self, image_path: str):
        """Open amazing image window."""
        viz_hdf5_imagery(image_path, ["ch6"])


class DataLoader(object):
    """Load the data from disk using tensorflow Dataset."""

    def __init__(self, image_reader: ImageReader, config: Dict[str, Any] = {}) -> None:
        """Create a DataLoader with some user config."""
        self.image_reader = image_reader
        self.config = config

    def create_dataset(self, metadata: Iterable[metadata.Metadata]) -> tf.data.Dataset:
        """Create a tensorflow Dataset base on the metadata and dataloader's config."""

        def gen():
            for md in metadata:
                image = self.image_reader.read(md.image_path, md.image_offset)
                yield tf.convert_to_tensor(image, dtype=tf.int64)

        return tf.data.Dataset.from_generator(gen, (tf.int64))
