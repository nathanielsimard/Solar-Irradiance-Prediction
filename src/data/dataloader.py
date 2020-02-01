from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np
import tensorflow as tf

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

    def __init__(self, channels=["ch1"], output_size=(64, 64)):
        """Default channel for image reading is ch1."""
        self.channels = channels
        self.output_size = output_size

    def read(
        self,
        image_path: str,
        image_offset: int,
        coordinates: Optional[metadata.Coordinates] = None,
    ) -> np.ndarray:
        """Read image with multiple channels from a compressed file.

        Args:
            image_path: The image location on disk.
            image_offset: The sample id, which image to read from the file.
            coordinates (Optional): The coordinates which the image will be centered.
                If provided, the image will be croped according to the output_size
                provided in the __init__ function.
        """
        try:
            with h5py.File(image_path) as file_reader:
                images = self._read_images(image_offset, file_reader)

                if coordinates is not None:
                    images = self._center_images(
                        images, image_offset, coordinates, file_reader
                    )

                return np.dstack(images)
        except OSError as e:
            raise InvalidImagePath(e)

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

    def _center_images(self, images, image_offset, coordinates, file_reader):
        return [
            self._center_image(image, image_offset, coordinates, file_reader)
            for image in images
        ]

    def _center_image(self, image, offset, coordinates, file_reader):
        center_pixel = self._coordinates_to_pixel(coordinates, file_reader, offset)

        start_x = center_pixel[0] - (self.output_size[0] // 2)
        start_y = center_pixel[1] - (self.output_size[1] // 2)

        end_x = start_x + self.output_size[0]
        end_y = start_y + self.output_size[1]

        return image[start_x:end_x, start_y:end_y]

    def _coordinates_to_pixel(self, coordinates, file_reader, offset):
        image_lat = fetch_hdf5_sample("lat", file_reader, offset)
        image_lon = fetch_hdf5_sample("lon", file_reader, offset)

        pixel_x = np.argmin(np.abs(image_lat - coordinates.latitude))
        pixel_y = np.argmin(np.abs(image_lon - coordinates.longitude))

        return pixel_x, pixel_y

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
