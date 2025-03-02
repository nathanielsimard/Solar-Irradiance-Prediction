import os
import pickle
from typing import Optional, Tuple

import h5py
import numpy as np

from src import logging
from src.data import metadata
from src.data.utils import fetch_hdf5_sample, viz_hdf5_imagery

logger = logging.create_logger(__name__)


class ImageNotCached(Exception):
    """Exception raised when image is not cached but was supposed to be."""

    pass


class InvalidImageOffSet(Exception):
    """Exception raised when offset isn't valid."""

    pass


class InvalidImageChannel(Exception):
    """Exception raised when channel isn't valid (valid channel: ch1, ch2, ch3, ch4, ch6)."""

    pass


class InvalidImagePath(Exception):
    """Exception when the path is not found."""

    pass


class CorruptedImage(Exception):
    """Exception when unable to read image from file."""

    pass


class ImageReader(object):
    """Read the images. Compression format is handle automaticly."""

    def __init__(
        self, channels=["ch1"], cache_dir=None, enable_caching=True, force_caching=False
    ):
        """Default channel for image reading is ch1.

        Args:
            channels: The channels to read from the file.
            cache_dir: Directory where cached image will be.
            enable_caching: Should we cache at all at this level.
            force_caching: Will throw an exception if the image is not cached.
        """
        self.channels = channels
        self.cache_dir = cache_dir
        self.enable_caching = enable_caching
        self.force_caching = force_caching
        self.cache_hits = 0
        self.cache_miss = 0

        if self.enable_caching and self.cache_dir is None:
            raise ValueError("Not cache dir provided, but cache was enabled")

    def read(
        self,
        image_path: str,
        image_offset: int,
        coordinates: metadata.Coordinates,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Read image with multiple channels from a compressed file.

        Args:
            image_path: The image location on disk.
            image_offset: The sample id, which image to read from the file.
            coordinates: The coordinates which the image will be centered.
            output_size (Optional): The image shape needed, if provided, the image
                will also be centered at coordinates when read.

        Return:
            Numpy array of shape (height, width, channel)

        Raises:
            InvalidImageChannel, InvalidImageOffSet, InvalidImagePath, CorruptedImage:
        """
        if self.enable_caching:
            cached_file = self._cache_file(
                image_path, image_offset, coordinates, output_size
            )
            try:
                image = self._load_cache_images(cached_file)
                self.cache_hits += 1
                return image
            except FileNotFoundError:
                self.cache_miss += 1
                logger.debug(f"Image {cached_file} not in cache")
                if self.cache_miss % 100 == 0:
                    logger.warning(f"{self.cache_hits} hits, {self.cache_miss} miss!")
                if self.force_caching:
                    raise ImageNotCached(
                        "Requested image not found in cache. Have you enabled 'force_caching' by mistake?"
                    )

        try:
            with h5py.File(image_path, "r") as file_reader:
                images = self._read_images(image_offset, file_reader)
                images = self._center_images(
                    images, image_offset, coordinates, file_reader, output_size
                )

                images = np.dstack(images)
                if self.enable_caching:
                    self._cache_images(images, cached_file)
                return images

        except OSError as e:
            raise InvalidImagePath(e)

    def _center_images(
        self, images, offset, coordinates, file_reader, output_size=None
    ):
        if output_size is None:
            return images

        image_lat = fetch_hdf5_sample("lat", file_reader, offset)
        image_lon = fetch_hdf5_sample("lon", file_reader, offset)

        return [
            center_image(image, image_lat, image_lon, coordinates, output_size)
            for image in images
        ]

    def _read_images(self, image_offset, file_reader):
        try:
            return [
                self._read_image_channel(image_offset, file_reader, channel)
                for channel in self.channels
            ]

        except ValueError as e:
            raise InvalidImageOffSet(e)
        except KeyError as e:
            raise InvalidImageChannel(e)

    def _read_image_channel(self, image_offset, file_reader, channel):
        sample = fetch_hdf5_sample(channel, file_reader, image_offset)
        if sample is None:
            raise CorruptedImage(
                "Unable to read image from file.\n"
                + f"    Filename: {file_reader.filename}\n"
                + f"    offset: {image_offset}\n"
                + f"    channel: {channel}\n"
            )

        norm_min = file_reader[channel].attrs.get("orig_min", None)
        norm_max = file_reader[channel].attrs.get("orig_max", None)

        scaled = ((sample - norm_min) / (norm_max - norm_min)) * 255
        return scaled.astype(np.uint8)

    def _cache_images(self, images, file_name):
        with open(file_name, "wb") as file:
            pickle.dump(images, file)

    def _load_cache_images(self, file_name) -> np.ndarray:
        with open(file_name, "rb") as file:
            return pickle.load(file)

    def _cache_file(
        self,
        image_path: str,
        image_offset: int,
        coordinates: metadata.Coordinates,
        output_size: Optional[Tuple[int, int]],
    ) -> str:
        name = os.path.basename(image_path)
        dir_name = (
            self.cache_dir
            + f"/{name}"
            + f"/{image_offset}"
            + f"/{coordinates}"
            + f"/{self.channels}"
        )
        file = dir_name + f"/{output_size}.pkl"
        os.makedirs(dir_name, exist_ok=True)
        return file

    def visualize(
        self, image_path: str, channel="ch1",
    ):
        """Open amazing image window."""
        viz_hdf5_imagery(image_path, [channel])


def center_image(
    image: np.ndarray,
    image_lat: np.ndarray,
    image_lon: np.ndarray,
    coordinates: metadata.Coordinates,
    output_size: Tuple[int, int],
):
    """Center the image at the coordinates.

    Args:
        image: 2D numpy array (height, width).
        image_lat: 2D numpy array (height, width) where each (i, j)
            correspond to the latitude of the image at (i, j).
        image_lon: 2D numpy array (height, width) where each (i, j)
            correspond to the longitude of the image at (i, j).
        coordinates: Coordinates of the center point.
        output_size: The size of the output image.
            If the size of too big, zeros-padding is applied.

    Return:
        2D numpy array of size 'output_size' centered at 'coordinates'.
    """
    center_pixel = _coordinates_to_pixel(coordinates, image_lat, image_lon)

    start_x = center_pixel[0] - (output_size[0] // 2)
    start_y = center_pixel[1] - (output_size[1] // 2)

    end_x = start_x + output_size[0]
    end_y = start_y + output_size[1]

    image_out_of_bound = (
        start_x < 0 or start_y < 0 or end_x > image.shape[0] or end_y > image.shape[1]
    )

    if image_out_of_bound:
        return _center_image_out_of_bound(
            image, start_x, end_x, start_y, end_y, output_size
        )

    return image[start_x:end_x, start_y:end_y]


def _center_image_out_of_bound(image, start_x, end_x, start_y, end_y, output_size):
    fixed_start_x = start_x if start_x >= 0 else 0
    fixed_start_y = start_y if start_y >= 0 else 0

    fixed_end_x = end_x if end_x <= image.shape[0] else image.shape[0]
    fixed_end_y = end_y if end_y <= image.shape[1] else image.shape[1]

    image_cropped = image[fixed_start_x:fixed_end_x, fixed_start_y:fixed_end_y]

    output_start_x = 0 if start_x >= 0 else -1 * start_x
    output_start_y = 0 if start_y >= 0 else -1 * start_y

    output_end_x = output_start_x + image_cropped.shape[0]
    output_end_y = output_start_y + image_cropped.shape[1]

    output = np.zeros(output_size)
    output[output_start_x:output_end_x, output_start_y:output_end_y] = image_cropped

    return output


def _coordinates_to_pixel(coordinates, image_lat, image_lon):
    pixel_x = np.argmin(np.abs(image_lat - coordinates.latitude))
    pixel_y = np.argmin(np.abs(image_lon - coordinates.longitude))

    return pixel_x, pixel_y
