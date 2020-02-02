from typing import Any, Dict, Iterable

import logging

import tensorflow as tf
import numpy as np

from pathlib import Path
from enum import IntEnum

from src.data import image, metadata
from src.data.image import CorruptedImage
import src.data.config as config
import src.data.clearskydata as csd


class AugmentedFeatures(IntEnum):
    """Mapping for the augmented features to the location in the tensor."""
    GHI_T = 0
    GHI_T_1h = 1
    GHI_T_3h = 2
    GHI_T_6h = 3
    SOLAR_TIME = 4


class DataLoader(object):
    """Load the data from disk using tensorflow Dataset.

    To load a batch of data, you can iterate over the tf.data.Dataset by batch.
    >>>dataset=dataset.batch(batch_size)
    """

    def __init__(
        self, image_reader: image.ImageReader, config: Dict[str, Any] = {}
    ) -> None:
        """Create a DataLoader with some user config.

        TODO: Describe what is going to be in the configuration.

        config["LOCAL_PATH"] = Allows overide of the base path on the server
                               to a local path. This will enable training on
                               the local machine.

        config["SKIP_MISSING"]= Will skip missing samples, just leaving a warning
                                instead of throwing an exception.
        config["ENABLE_META"] = Will enable outputing meta data along with the other data.

        config["CROP_SIZE"] = Size of the crop image arround the center. None will return the 
                              whole image.

        """
        self.image_reader = image_reader
        self.config = config
        self.skip_missing = False
        self.local_path = None
        self.metadata = None
        self.enable_meta = False
        self.crop_size = (64, 64)  # Default for now, we should add a parameter

        if "SKIP_MISSING" in config:
            self.skip_missing = config["SKIP_MISSING"]
        if "LOCAL_PATH" in self.config:
            self.local_path = config["LOCAL_PATH"]
        if "ENABLE_META" in self.config:
            self.enable_meta = config["ENABLE_META"]
        if "CROP_SIZE" in self.config:
            self.crop_size = config["CROP_SIZE"]

    def _prepare_meta(self, md: metadata.Metadata):
        # TODO: Add other meta information, such as local solar time.
        meta = np.zeros(len(AugmentedFeatures))
        clearsky_values = csd.get_clearsky_values(md.coordinates, md.datetime)
        meta[0:len(clearsky_values)] = clearsky_values

        return tf.convert_to_tensor(meta)

    def _transform_image_path(self, original_path):
        """Transforms a supplied path on "helios" to a local path."""
        if self.local_path is not None:
            # "/home/raphael/MILA/ift6759/project1_data/hdf5v7_8bit/"
            return str(Path(self.local_path + "/" + Path(original_path).name))
        else:
            return original_path

    # I moved the generator outside the create_dataset function in order to
    # be able to debug it! I think it should stay here as I want to be able
    # to step in the code if something goes wrong.

    def gen(self):
        """Generator for images."""
        for md in self.metadata:
            logging.info(str(md))
            try:
                image = self.image_reader.read(
                    self._transform_image_path(md.image_path),
                    md.image_offset,
                    md.coordinates,
                )
            except CorruptedImage as e:
                if self.skip_missing:
                    logging.warning("Skipping corrupted image:" + str(md))
                    continue
                else:
                    raise e  # Not handled!

            data = tf.convert_to_tensor(image, dtype=tf.float32)
            target = tf.constant(
                [
                    _target_value(md.target_ghi),
                    _target_value(md.target_ghi_1h),
                    _target_value(md.target_ghi_3h),
                    _target_value(md.target_ghi_6h),
                ]
            )
            output = (data, target)
            if self.enable_meta:
                meta = self._prepare_meta(md)
                output = output + (meta, )
            yield output

    def create_dataset(self, metadata: Iterable[metadata.Metadata]) -> tf.data.Dataset:
        """Create a tensorflow Dataset base on the metadata and dataloader's config.

        Targets are optional in Metadata. If one is missing, set it to zero.
        """
        output_shape = (tf.float32, tf.float32)
        if self.enable_meta:
            output_shape = output_shape + (tf.float32, )
        if self.metadata is not None:
            raise ValueError("Create_dataset can only be called once per instance")
        self.metadata = metadata
        return tf.data.Dataset.from_generator(self.gen, output_shape)


def _target_value(target):
    if target is None:
        return 0
    return target
