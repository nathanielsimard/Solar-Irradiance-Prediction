from typing import Any, Dict, Iterable

import tensorflow as tf

from src.data import image, metadata


class DataLoader(object):
    """Load the data from disk using tensorflow Dataset.

    To load a batch of data, you can iterate over the tf.data.Dataset by batch.
    >>>dataset=dataset.batch(batch_size)
    """

    def __init__(
        self, image_reader: image.ImageReader, config: Dict[str, Any] = {}
    ) -> None:
        """Create a DataLoader with some user config."""
        self.image_reader = image_reader
        self.config = config

    def create_dataset(self, metadata: Iterable[metadata.Metadata]) -> tf.data.Dataset:
        """Create a tensorflow Dataset base on the metadata and dataloader's config.

        Targets are optional in Metadata. If one is missing, set it to zero.
        """

        def gen():
            for md in metadata:
                image = self.image_reader.read(
                    md.image_path, md.image_offset, md.coordinates
                )
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

        return tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))


def _target_value(target):
    if target is None:
        return 0
    return target
