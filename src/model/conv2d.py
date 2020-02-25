from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "Conv2D"
NAME_CLEARSKY = "Conv2DClearsky"
NAME_CLEARSKY_V2 = "Conv2DClearskyV2"


class CNN2D(base.Model):
    """Create Conv2D model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.min_max_scaling_images()
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()

        self.conv1 = self._convolution_step((5, 5), 64)
        self.conv2 = self._convolution_step((3, 3), 128)
        self.conv3 = self._convolution_step((3, 3), 256)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(1)

    def call(self, data: Tuple[tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = data[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv2d_1 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_2 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_3 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling2D(pool_size=(2, 2))

        return Sequential([conv2d_1, conv2d_2, conv2d_3, max_pool])

    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 0.01
        config.features = [dataloader.Feature.image, dataloader.Feature.target_ghi]

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda image, target_ghi: (
                self.scaling_image.normalize(image),
                self._preprocess_target(target_ghi),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]


class CNN2DClearsky(base.Model):
    """Create Conv2D model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME_CLEARSKY)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()
        self.conv1 = self._convolution_step((5, 5), 128)
        self.conv2 = self._convolution_step((3, 3), 256)
        self.dropout2 = Dropout(0.1)
        self.conv3 = self._convolution_step((3, 3), 256)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(4)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        _, meta, x = data

        x = self.conv1(x)
        x = self.dropout2(self.conv2(x), training)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        z = tf.concat([x, meta], 1)  # Late combining of the metadata.
        x = self.d4(z)
        # x = self.d4(x)
        x = self.d5(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv2d_1 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_2 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_3 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling2D(pool_size=(2, 2))

        return Sequential([conv2d_1, conv2d_2, conv2d_3, max_pool])

    def config(self, dry_run=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 1
        config.features = [
            dataloader.Feature.target_ghi,
            dataloader.Feature.metadata,
            dataloader.Feature.image,
            dataloader.Feature.target_ghi,
        ]

        if dry_run:
            config.error_strategy = dataloader.ErrorStrategy.stop
        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        def preprocess(target_ghi_dummy, metadata, image, target_ghi):
            image = self.scaling_image.normalize(image)
            metadata = self.scaling_ghi.normalize(metadata)
            target_ghi = self.scaling_ghi.normalize(target_ghi)

            return target_ghi_dummy, metadata, image, target_ghi

        return dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class CNN2DClearskyV2(base.Model):
    # This one uses the architecture proposed by
    # Johan Mathe, Nina Miolane, Nicolas Sebastien and Jeremie Lequeux
    # https://arxiv.org/pdf/1902.01453.pdf
    """Create Conv2D model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()
        # Conv2D(channels, kernel_size=kernel_size, activation="relu")
        self.conv1 = Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.dropout2 = Dropout(0.1)
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(128, kernel_size=(3, 3), activation="relu")
        self.conv4 = Conv2D(128, kernel_size=(3, 3), activation="relu")
        self.conv5 = Conv2D(256, kernel_size=(3, 3), activation="relu")
        self.conv6 = Conv2D(256, kernel_size=(3, 3), activation="relu")
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(4)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        _, meta, x = data

        x = self.conv1(x)
        x = self.dropout2(self.conv2(x), training)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool2(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        z = tf.concat([x, meta], 1)  # Late combining of the metadata.
        x = self.d4(z)
        # x = self.d4(x)
        x = self.d5(x)

        return x

    def config(self, training=False, dry_run=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 1
        config.features = [
            dataloader.Feature.target_ghi,
            dataloader.Feature.metadata,
            dataloader.Feature.image,
            dataloader.Feature.target_ghi,
        ]
        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore
        if dry_run:
            config.error_strategy = dataloader.ErrorStrategy.stop
        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        def preprocess(target_ghi_dummy, metadata, image, target_ghi):
            image = self.scaling_image.normalize(image)
            metadata = self.scaling_ghi.normalize(metadata)
            target_ghi = self.scaling_ghi.normalize(target_ghi)

            return target_ghi_dummy, metadata, image, target_ghi

        return dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
