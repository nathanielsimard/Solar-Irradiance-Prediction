import logging
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import src.data.clearskydata as csd
from src.data import image
from src.data.image import CorruptedImage, InvalidImagePath
from src.data.metadata import Metadata


class Feature(Enum):
    """Feature which the dataloader can load."""

    image = "image"
    target_ghi = "target_ghi"
    metadata = "metadata"


class UnregognizedFeature(Exception):
    """Exception raised when parsing config."""

    def __init__(self, feature: str):
        """Create an error message from the unrecognized feature."""
        super().__init__(
            f"Feature '{feature}' is unrecognized.\n"
            + f"Valid features are {Feature.image},"
            + f"{Feature.target_ghi} and {Feature.metadata}."
        )


class Config:
    """Configuration available to the dataloader."""

    def __init__(
        self,
        local_path: Optional[str] = None,
        skip_missing: bool = False,
        crop_size: Tuple[int, int] = (64, 64),
        features: List[Feature] = [Feature.image, Feature.target_ghi],
        channels: List[str] = ["ch1"],
    ):
        """All configurations are optional with default values.

        Args:
            local_path: Can overrite the root path of each images.
            skip_missing: When no image is found, don't load the data.
            crop_size: Image sized needed.
            features: List of features needed.
            channels: List of channels needed.
        """
        self.local_path = local_path
        self.skip_missing = skip_missing
        self.crop_size = crop_size
        self.features = features
        self.channels = channels


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
        metadata: Iterable[Metadata],
        image_reader: image.ImageReader,
        config: Config = Config(),
    ):
        """Load the config with the image_reader from the metadata."""
        self.metadata = metadata
        self.image_reader = image_reader
        self.config = config

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
        for metadata in self.metadata:
            logging.info(str(metadata))

            yield tuple(
                [self._readers[feature](metadata) for feature in self.config.features]
            )

    def _read_target(self, metadata: Metadata) -> tf.Tensor:
        return tf.constant(
            [
                _target_value(metadata.target_ghi),
                _target_value(metadata.target_ghi_1h),
                _target_value(metadata.target_ghi_3h),
                _target_value(metadata.target_ghi_6h),
            ]
        )

    def _read_image(self, metadata: Metadata) -> tf.Tensor:
        image_path = self._transform_image_path(metadata.image_path)
        try:
            image = self.image_reader.read(
                image_path,
                metadata.image_offset,
                metadata.coordinates,
                self.config.crop_size,
            )
        except (CorruptedImage, InvalidImagePath) as e:
            if not self.config.skip_missing:
                raise e
            logging.warning(f"Error while reading image, skipping : {e}")

        return tf.convert_to_tensor(image, dtype=tf.float32)

    def _read_metadata(self, metadata: Metadata) -> tf.Tensor:
        meta = np.zeros(len(MetadataFeatureIndex))
        clearsky_values = csd.calculate_clearsky_values(
            metadata.coordinates, metadata.datetime
        )
        meta[0 : len(clearsky_values)] = clearsky_values

        return tf.convert_to_tensor(meta)

    def _transform_image_path(self, original_path):
        """Transforms a supplied path on "helios" to a local path."""
        if self.config.local_path is None:
            return original_path

        return str(Path(self.config.local_path + "/" + Path(original_path).name))


def create_dataset(
    metadata: Iterable[Metadata], config: Union[Dict[str, Any], Config] = Config()
) -> tf.data.Dataset:
    """Create a tensorflow Dataset base on the metadata and dataloader's config.

    Targets are optional in Metadata. If one is missing, set it to zero.
    To load a batch of data, you can iterate over the tf.data.Dataset by batch.
    >>>dataset=dataset.batch(batch_size)
    """
    if isinstance(config, Dict):
        config = parse_config(config)

    image_reader = image.ImageReader(channels=config.channels)

    dataloader = DataLoader(metadata, image_reader, config=config)

    features_type = tuple(len(config.features) * [tf.float32])
    return tf.data.Dataset.from_generator(dataloader.generator, features_type)


def parse_config(config: Dict[str, Any] = {}) -> Config:
    """Parse the user config.

    TODO: Describe what is going to be in the configuration.

    config["LOCAL_PATH"]    = Allows overide of the base path on the server
                              to a local path. This will enable training on
                              the local machine.

    config["SKIP_MISSING"]  = Will skip missing samples, just leaving a warning
                              instead of throwing an exception.

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

    skip_missing = _read_config(config, "SKIP_MISSING", False)
    local_path = _read_config(config, "LOCAL_PATH", None)
    crop_size = _read_config(config, "CROP_SIZE", (64, 64))
    channels = _read_config(config, "CHANNELS", ["ch1"])

    return Config(
        local_path=local_path,
        skip_missing=skip_missing,
        crop_size=crop_size,
        features=features,
        channels=channels,
    )


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


def _target_value(target):
    if target is None:
        return 0
    return target
