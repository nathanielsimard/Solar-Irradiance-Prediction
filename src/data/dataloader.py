from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import src.data.clearskydata as csd
from src import logging
from src.data import image
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


class ErrorStrategy(Enum):
    """How error are handled by the dataloader."""

    skip = "skip"  # Ignore the sample with missing data, and proceed to the next.
    ignore = "ignore"  # Return a black image when data is mising
    stop = "stop"  # Stop code execution.


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
    ):
        """All configurations are optional with default values.

        Args:
            local_path: Can overrite the root path of each images.
            error_strategy: How to handle errors.
            force_caching: Option to skip non cached images.
            crop_size: Image sized needed.
            features: List of features needed.
            channels: List of channels needed.
            image_cache_dir: Where the crop images will be cached.
            num_images: Total number of images.
                If more than 1, images from the past are goin to be included.
            time_interval_min: Number of minutes between images.
                If num_images is 1, this has no effets.
            ratio: proportion of the data we want.
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


class MetadataFeatureIndex(IntEnum):
    """Mapping for the augmented features to the location in the tensor."""

    GHI_T = 0
    GHI_T_1h = 1
    GHI_T_3h = 2
    GHI_T_6h = 3
    SOLAR_TIME = 4


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
        self.csd = csd.Clearsky()

        self._readers = {
            Feature.image: self._read_image,
            Feature.target_ghi: self._read_target,
            Feature.metadata: self._read_metadata,
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
            try:
                yield tuple(
                    [
                        self._readers[feature](metadata)
                        for feature in self.config.features
                    ]
                )
            except AttributeError as e:
                # This is clearly unhandled! We want a crash here!
                raise e
                # TODO: We should list the handled exceptions here, and do
                # a stack trace if we encounter something really wrong.
            except Exception as e:
                if self.config.error_strategy == ErrorStrategy.stop:
                    logger.error(f"Error while generating data, stopping : {e}")
                    raise e
                logger.debug(f"Error while generating data, skipping : {e}")

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
        # We should only catch here exceptions that are safe to ignore.
        except (InvalidImageChannel, InvalidImageOffSet, CorruptedImage) as e:
            if self.config.error_strategy != ErrorStrategy.ignore:
                raise e  # Skip
            logger.debug(f"Error while generating data, ignoring : {e}")
            output_shape = list(self.config.crop_size) + [len(self.config.channels)]
            return tf.convert_to_tensor(np.zeros(output_shape))
        except (ImageNotCached) as e:
            if self.config.force_caching:
                raise e  # Skip

        except (Exception) as e:
            raise e  # Some error require immediate attention!

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
                logger.debug(f"Error while generating past images, ignoring : {e}")
                images.append(np.zeros(shape))

        return images

    def _read_metadata(self, metadata: Metadata) -> tf.Tensor:
        meta = np.zeros(len(MetadataFeatureIndex))
        clearsky_values = self.csd.calculate_clearsky_values(
            metadata.coordinates, metadata.datetime
        )
        """This reader will read all information that is not contained
        in the image. It will allow to train using the computed clearsky values.

        In order to use it, the configuration must say the we use this reader.

        It will yield a single vector containing all values side by side for
        this sample. (T, T+1, T+3, T+6 )
        """
        meta[0 : len(clearsky_values)] = clearsky_values

        return tf.convert_to_tensor(meta)

    def _target_value(self, target):
        if target is not None:
            return target

        if self.config.error_strategy == ErrorStrategy.ignore:
            return 0.0

        raise MissingTargetException()

    def _transform_image_path(self, original_path):
        """Transforms a supplied path on "helios" to a local path."""
        if self.config.local_path is None:
            return original_path

        return str(Path(self.config.local_path + "/" + Path(original_path).name))


def create_dataset(
    metadata: Callable[[], Iterable[Metadata]],
    config: Union[Dict[str, Any], DataloaderConfig] = DataloaderConfig(),
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
    )
    dataloader = DataLoader(metadata, image_reader, config=config)

    return tf.data.Dataset.from_generator(dataloader.generator, features_type)


def create_generator(
    metadata: Callable[[], Iterable[Metadata]],
    config: Union[Dict[str, Any], DataloaderConfig] = DataloaderConfig(),
) -> tf.data.Dataset:
    """Create a generator that will to the dataloader work. Will be used for debugging.

    Might be scrapped later on.

    Targets are optional in Metadata. If one is missing, set it to zero.
    To load a batch of data, you can iterate over the tf.data.Dataset by batch.
    >>>dataset=dataset.batch(batch_size)
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
