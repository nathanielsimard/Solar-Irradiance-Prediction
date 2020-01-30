from typing import Any, Dict, Iterable

import numpy as np
import tensorflow as tf

from src.data import metadata


class ImageReader(object):
    def read(self, image_path: str, image_offset: int) -> np.ndarray:
        pass


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
