from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import src.data.clearskydata as csd
from src import logging
from src.data import image
from src.data.config import Coordinates, Station
from src.data.image import (
    CorruptedImage,
    ImageNotCached,
    InvalidImageChannel,
    InvalidImageOffSet,
)
from src.data.metadata import Metadata

logger = logging.create_logger(__name__)


class Feature(Enum):
    """Feature which the dataloader can load."""

    image = "image"
    target_ghi = "target_ghi"
    metadata = "metadata"
    target_cloud = "target_cloud"


class ErrorStrategy(Enum):
    """How error are handled by the dataloader.

    skip: Ignore the sample with missing data, and proceed to the next.
    ignore: Return a black image when data is mising
    stop: Stop code execution.

    Note:
        Some errors might have their own error configuration
        like how missing past images are handled.

    """

    skip = "skip"
    ignore = "ignore"
    stop = "stop"


class UnregognizedFeature(Exception):
    """Exception raised when parsing config."""

    def __init__(self, feature: str):
        """Create an error message from the unrecognized feature."""
        possible_values = [e.value for e in Feature]
        super().__init__(
            f"Feature '{feature}' is unrecognized.\n"
            + f"Valid features are {possible_values}."
        )


class UnregognizedErrorStrategy(Exception):
    """Exception raised when parsing config."""

    def __init__(self, error_strategy: str):
        """Create an error message from the unrecognized error strategy."""
        possible_values = [e.value for e in ErrorStrategy]
        super().__init__(
            f"Error Stategy '{error_strategy}' is unrecognized.\n"
            + f"Valid error strategies are {possible_values}."
        )


class MissingTargetException(Exception):
    """Exception raised when reading targets."""

    def __init__(self):
        """Create an error message."""
        super().__init__(f"Target is missing.")


class DataloaderConfig:
    """Configuration available to the dataloader."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        error_strategy=ErrorStrategy.skip,
        force_caching=False,
        crop_size: Tuple[int, int] = (64, 64),
        features: List[Feature] = [Feature.image, Feature.target_ghi],
        channels: List[str] = ["ch1"],
        image_cache_dir="/tmp",
        num_images=1,
        time_interval_min=15,
        ratio=1,
        target_datetimes=None,
        stations: Dict[Station, Coordinates] = None,
        precompute_clearsky=False,
        skip_missing_past_images=False,
        filter_night=True,
    ):
        """All configurations are optional with default values.

        Args:
            local_path: Can override the root path of each images.
            error_strategy: How to handle errors.
            force_caching: Option to skip non cached images.
            crop_size: Image size needed.
            features: List of features needed.
                The features will be provided in order.
            channels: List of channels needed.
            image_cache_dir: Where the cropped images will be cached.
            num_images: Total number of images.
                If more than 1, images from the past are going to be included.
            time_interval_min: Number of minutes between images.
                If num_images is 1, this has no effect.
            ratio: proportion of the data we want.
            target_datetimes: list of target datetimes for clearsky caching
            stations: list of station where to pre-compute
            precompute_clearsky: Will pre-compute clearsky values if set to true.
            skip_missing_past_images: if past image is missing, skip.
            filter_night: if metadata are to be filtered when it is night time.
        """
        self.local_path = local_path
        self.error_strategy = error_strategy
        self.crop_size = crop_size
        self.features = features
        self.channels = channels
        self.force_caching = force_caching
        self.image_cache_dir = image_cache_dir
        self.num_images = num_images
        self.time_interval_min = time_interval_min
        self.ratio = ratio
        self.target_datetimes = target_datetimes
        self.stations = stations
        self.precompute_clearsky = precompute_clearsky
        self.skip_missing_past_images = skip_missing_past_images
        self.filter_night = filter_night

    def __str__(self):
        """Return nice string representation of the config."""
        string = "{\n"
        attributes = vars(self)
        for key, value in attributes.items():
            string += f"  {key}: {value}\n"
        return string + "}\n"


class MetadataFeatureIndex(IntEnum):
    """Mapping for the augmented features to the location in the tensor."""

    GHI_T = 0
    GHI_T_1h = 1
    GHI_T_3h = 2
    GHI_T_6h = 3


class DataLoader(object):
    """Load the data from disk using generator."""

    def __init__(
        self,
        metadata: Callable[[], Iterable[Metadata]],
        image_reader: image.ImageReader,
        config: DataloaderConfig = DataloaderConfig(),
    ):
        """Load the config with the image_reader from the metadata."""
        self.metadata = metadata
        self.image_reader = image_reader
        self.config = config
        enable_clearsky_caching = False

        if config.precompute_clearsky:
            enable_clearsky_caching = True

        self.csd = csd.Clearsky(enable_caching=enable_clearsky_caching)

        if config.precompute_clearsky:
            self.csd._precompute_clearsky_values(
                config.target_datetimes, config.stations
            )

        self.ok = 0
        self.skipped = 0

        self._readers = {
            Feature.image: self._read_image,
            Feature.target_ghi: self._read_target,
            Feature.metadata: self._read_metadata,
            Feature.target_cloud: self._read_cloudiness,
        }

    def generator(self):
        """Generate features based on config.

        Return:
            Generator where each yield element is a tuple.
            Each index of the element corresponds to one feature in the configuration.
            The order is kept.
        """
        for metadata in self.metadata():
            logger.debug(metadata)

            if self.config.filter_night and metadata.night_time:
                continue

            try:
                output = [
                    self._readers[feature](metadata)
                    for feature in reversed(self.config.features)
                ]
                self.ok += 1
                yield tuple(reversed(output))
            except AttributeError as e:
                logger.error(f"Error while generating data, stopping : {e}")
                raise e
            except Exception as e:
                if self.config.error_strategy == ErrorStrategy.stop:
                    logger.error(f"Error while generating data, stopping : {e}")
                    raise e

                logger.debug(f"Error while generating data, skipping : {e}")
                self.skipped += 1
                if (self.skipped % 1000) == 0:
                    logger.warning(f"{self.skipped} skipped, {self.ok} ok.")

    def _read_cloudiness(self, metadata: Metadata) -> tf.Tensor:
        return tf.convert_to_tensor(
            [
                self._convert_cloud_to_oneHot(
                    self._target_cloud(metadata.target_cloudiness)
                ),
                self._convert_cloud_to_oneHot(
                    self._target_cloud(metadata.target_cloudiness_1h)
                ),
                self._convert_cloud_to_oneHot(
                    self._target_cloud(metadata.target_cloudiness_3h)
                ),
                self._convert_cloud_to_oneHot(
                    self._target_cloud(metadata.target_cloudiness_6h)
                ),
            ]
        )

    def _convert_cloud_to_oneHot(self, cloud):
        one_hot = {
            "night": np.array([1, 0, 0, 0, 0]),
            "cloudy": np.array([0, 1, 0, 0, 0]),
            "slightly cloudy": np.array([0, 0, 1, 0, 0]),
            "clear": np.array([0, 0, 0, 1, 0]),
            "variable": np.array([0, 0, 0, 0, 1]),
        }

        try:
            return one_hot[cloud]
        except KeyError:
            return [0, 0, 0, 0, 0]

    def _read_target(self, metadata: Metadata) -> tf.Tensor:
        return tf.convert_to_tensor(
            [
                self._target_value(metadata.target_ghi),
                self._target_value(metadata.target_ghi_1h),
                self._target_value(metadata.target_ghi_3h),
                self._target_value(metadata.target_ghi_6h),
            ],
            dtype=tf.float32,
        )

    def _read_image(self, metadata: Metadata) -> tf.Tensor:
        try:
            current_image_path = metadata.image_paths[-1]
            current_image_offset = metadata.image_offsets[-1]
            past_image_paths = metadata.image_paths[:-1]
            past_image_offsets = metadata.image_offsets[:-1]

            current_image = self.image_reader.read(
                self._transform_image_path(current_image_path),
                current_image_offset,
                metadata.coordinates,
                self.config.crop_size,
            )

            past_images = self._read_past_images(
                past_image_paths,
                past_image_offsets,
                metadata.coordinates,
                current_image.shape,
            )

            if len(past_images) > 0:
                images = np.stack(past_images + [current_image])
            else:
                images = current_image

            return tf.convert_to_tensor(images, dtype=tf.float32)
        except (InvalidImageChannel, InvalidImageOffSet, CorruptedImage) as e:
            if self.config.error_strategy != ErrorStrategy.ignore:
                # The item will either be skipped or the pipeline will stopped.
                # The exception cannot be handled here.
                raise e

            logger.debug(f"Error while generating data, ignoring : {e}")
            output_shape = list(self.config.crop_size) + [len(self.config.channels)]
            if self.config.num_images > 1:
                output_shape = [self.config.num_images] + output_shape
            return tf.convert_to_tensor(np.zeros(output_shape))
        except ImageNotCached as e:
            if self.config.force_caching:
                logger.debug(f"Error while generating data, skipping : {e}")
                raise e

        except Exception as e:
            # Some unknown error, require immediate attention!
            raise e

    def _read_past_images(self, image_paths, image_offsets, coordinates, shape):
        images = []
        for image_path, image_offset in zip(image_paths, image_offsets):
            try:
                image = self.image_reader.read(
                    self._transform_image_path(image_path),
                    image_offset,
                    coordinates,
                    self.config.crop_size,
                )
                images.append(image)
            except Exception as e:
                if self.config.skip_missing_past_images:
                    logger.debug(f"Error while generating past images, skipping : {e}")
                    raise e
                logger.debug(f"Error while generating past images, ignoring : {e}")
                images.append(np.zeros(shape))

        return images

    def _read_metadata(self, metadata: Metadata) -> tf.Tensor:
        meta = np.zeros(len(MetadataFeatureIndex))
        """This reader will allow to train using the computed clearsky values.

        The metadata feature must be included in the config.

        It will yield a single vector containing all values side by side for
        this sample. (T, T+1, T+3, T+6 )
        """
        clearsky_values = self.csd.calculate_clearsky_values(
            metadata.coordinates, metadata.datetime
        )
        meta[0 : len(clearsky_values)] = clearsky_values

        return tf.convert_to_tensor(meta, dtype=tf.float32)

    def _clearsky_value(self, value):
        if value is not None:
            return value

        return -1

    def _target_value(self, target):
        if target is not None:
            return target

        if self.config.error_strategy == ErrorStrategy.ignore:
            return 0.0

        raise MissingTargetException()

    def _target_cloud(self, target):
        if target is not None:
            return target

        if self.config.error_strategy == ErrorStrategy.ignore:
            return "variable"

        raise MissingTargetException()

    def _transform_image_path(self, original_path):
        """Transforms a supplied path on "helios" to a local path."""
        if self.config.local_path is None:
            return original_path

        return str(Path(self.config.local_path + "/" + Path(original_path).name))


def create_dataset(
    metadata: Callable[[], Iterable[Metadata]],
    config: Union[Dict[str, Any], DataloaderConfig] = DataloaderConfig(),
    target_datetimes=None,
    stations: Dict[Station, Coordinates] = None,
    enable_image_cache=True,
) -> tf.data.Dataset:
    """Create a tensorflow Dataset base on the metadata and dataloader's config.

    Targets are optional in Metadata. If one is missing, set it to zero.
    To load a batch of data, you can iterate over the tf.data.Dataset by batch.
    >>>dataset=dataset.batch(batch_size)
    """
    if isinstance(config, Dict):
        config = parse_config(config)

    features_type = tuple(len(config.features) * [tf.float32])
    image_reader = image.ImageReader(
        channels=config.channels,
        cache_dir=config.image_cache_dir,
        force_caching=config.force_caching,
        enable_caching=enable_image_cache,
    )
    dataloader = DataLoader(metadata, image_reader, config=config)

    return tf.data.Dataset.from_generator(dataloader.generator, features_type).prefetch(
        tf.data.experimental.AUTOTUNE
    )


def create_generator(
    metadata: Callable[[], Iterable[Metadata]],
    config: Union[Dict[str, Any], DataloaderConfig] = DataloaderConfig(),
):
    """Creates a data generator for the dataloader.

    Alternative to tf.data.DataSet, but tf version is prepared.
    """
    if isinstance(config, Dict):
        config = parse_config(config)

    image_reader = image.ImageReader(
        channels=config.channels,
        cache_dir=config.image_cache_dir,
        force_caching=config.force_caching,
    )
    return DataLoader(metadata, image_reader, config=config).generator()


def parse_config(config: Dict[str, Any] = {}) -> DataloaderConfig:
    """Parse the user config.

    TODO: Describe what is going to be in the configuration.

    config["LOCAL_PATH"]    = Allows overide of the base path on the server
                              to a local path. This will enable training on
                              the local machine.

    config["ERROR_STATEGY"] = How error are handled by the dataloader.
                              Supported values are ["skip", "stop", "ignore"]

    config["CROP_SIZE"]     = Size of the crop image arround the center. None will return the
                              whole image.

    config["FEATURES"]        = A list of the each element needed in the features.
                              The dataset will return each element in a tuple in order.
                              Supported formats are : ["image", "target_ghi", "metadata"].
                              Defaults are ["image", "target_ghi"]

    config["CHANNELS"]      = Image channels to read.
                              Supported channels are ["ch1", "ch2", "ch3", "ch4", "ch6"]
                              Default is ["ch1"]
    """
    features = _read_config(
        config, "FEATURES", [Feature.image.value, Feature.target_ghi.value],
    )
    features = _parse_features(features)

    error_strategy = _read_config(config, "ERROR_STATEGY", ErrorStrategy.skip.value)
    error_strategy = _parse_error_strategy(error_strategy)

    local_path = _read_config(config, "LOCAL_PATH", None)
    crop_size = _read_config(config, "CROP_SIZE", (64, 64))
    channels = _read_config(config, "CHANNELS", ["ch1"])

    return DataloaderConfig(
        local_path=local_path,
        error_strategy=error_strategy,
        crop_size=crop_size,
        features=features,
        channels=channels,
    )


def _parse_error_strategy(error_strategy) -> ErrorStrategy:
    try:
        return ErrorStrategy(error_strategy)
    except ValueError:
        raise UnregognizedErrorStrategy(error_strategy)


def _parse_features(features) -> List[Feature]:
    parsed_features = []
    for feature in features:
        try:
            parsed_features.append(Feature(feature))
        except ValueError:
            raise UnregognizedFeature(feature)
    return parsed_features


def _read_config(config, key, default):
    if key not in config:
        return default
    return config[key]
