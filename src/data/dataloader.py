from typing import Any, Dict, Iterable

import tensorflow as tf
from pathlib import Path

from src.data import image, metadata


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

        """
        self.image_reader = image_reader
        self.config = config
        self.skip_missing = False
        self.local_path = None
        self.metadata = None

        if "SKIP_MISSING" in config:
            self.skip_missing = config["SKIP_MISSING"]
        if "LOCAL_PATH" in self.config:
            self.local_path = config["LOCAL_PATH"]

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
            image = self.image_reader.read(
                self._transform_image_path(md.image_path), md.image_offset
            )
            if image.size == 1:
                # No image was returned. We should skip for training.
                if self.skip_missing:
                    # TODO: Add logging code here!
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
        if self.metadata is not None:
            raise ValueError("Create_dataset can only be called once per instance")
        self.metadata = metadata
        return tf.data.Dataset.from_generator(self.gen, (tf.float32, tf.float32))


def _target_value(target):
    if target is None:
        return 0
    return target
