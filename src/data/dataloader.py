from typing import Any, Dict, Iterable

import h5py
import numpy as np
import tensorflow as tf
from enum import Enum
from pathlib import Path

from src.data import metadata
from src.data.utils import fetch_hdf5_sample, viz_hdf5_imagery


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

        return np.dstack(self._read_images(image_offset, file_reader))

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

    def visualize(
        self, image_path: str, channel="ch1",
    ):
        """Open amazing image window."""
        viz_hdf5_imagery(image_path, [channel])


class DataLoader(object):
    """Load the data from disk using tensorflow Dataset.

    To load a batch of data, you can iterate over the tf.data.Dataset by batch.
    >>>dataset=dataset.batch(batch_size)
    """

    def __init__(self, image_reader: ImageReader, config: Dict[str, Any] = {}) -> None:
        """Create a DataLoader with some user config.
        
        TODO: Describe what is going to be in the configuration.

        config["LOCAL_PATH"] = Allows overide of the base path on the server
                               to a local path. This will enable training on
                               the local machine.

        config["SKIP_MISSING"]= Will skip missing samples, just leaving a warning
                                instead of throwing an exception.

        """
        self.image_reader = image_reader
        self.config = config
        self.skip_missing = False

        if self.Parameters.SKIP_MISSING.name in config:
            self.skip_missing = config[self.Parameters.SKIP_MISSING.name]

    class Parameters(Enum):
        LOCAL_PATH = "LOCAL_PATH"
        SKIP_MISSING = "SKIP_MISSING"

    def _transform_image_path(self, original_path):
        """Transforms a supplied path on "helios" to a local path.
        """
        if DataLoader.Parameters.LOCAL_PATH.name in self.config:
            basedir = self.config[DataLoader.Parameters.LOCAL_PATH.name] #"/home/raphael/MILA/ift6759/project1_data/hdf5v7_8bit/"
            return str(Path(basedir + "/" + Path(original_path).name))
        else:
            return original_path        
    # I moved the generator outside the create_dataset function in order to 
    # be able to debug it! I think it should stay here as I want to be able 
    # to step in the code if something goes wrong.
    
    def gen(self):
        for md in self.metadata:
            image = self.image_reader.read(self._transform_image_path(md.image_path), md.image_offset)
            if image.size==1:
                #No image was returned. We should skip for training.
                if self.skip_missing:
                    #TODO: Add logging code here!
                    continue
            data = tf.convert_to_tensor(image, dtype=tf.float32)
            target = tf.constant(
                [
                    _target_value(md.target_ghi),
                    _target_value(md.target_ghi_1h),
                    _target_value(md.target_ghi_3h),
                    _target_value(md.target_ghi_6h),
                ]
            )
            yield (data, target)
            
    def create_dataset(self, metadata: Iterable[metadata.Metadata]) -> tf.data.Dataset:
        """Create a tensorflow Dataset base on the metadata and dataloader's config.

        Targets are optional in Metadata. If one is missing, set it to zero.
        """
        self.metadata = metadata
        return tf.data.Dataset.from_generator(self.gen, (tf.float32, tf.float32))


def _target_value(target):
    if target is None:
        return 0
    return target
