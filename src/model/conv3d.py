from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, Flatten, MaxPooling3D, Dropout
from tensorflow.keras.models import Sequential

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "Conv3D"


class CNN3D(base.Model):
    """Create Conv3D model."""

    def __init__(self, num_images=8):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.num_images = num_images

        self.conv1 = self._convolution_step((1, 5, 5), 64)
        self.conv2 = self._convolution_step((1, 3, 3), 128)
        self.conv3 = self._convolution_step((1, 3, 3), 256)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(4)

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
        conv3d_1 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        conv3d_2 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        conv3d_3 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling3D(pool_size=(1, 2, 2))

        return Sequential([conv3d_1, conv3d_2, conv3d_3, max_pool])

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.ratio = 1
        config.time_interval_min = 30
        config.features = [dataloader.Feature.image, dataloader.Feature.target_ghi]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda image, target_ghi: (self.scaling_image.normalize(image), target_ghi,)
        )


class CNN3D_Clearsky(base.Model):
    """Create Conv3D model."""

    def __init__(self, num_images=4):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.num_images = num_images

        self.conv1 = self._convolution_step((1, 5, 5), 64)
        self.conv2 = self._convolution_step((1, 3, 3), 128)
        self.conv3 = self._convolution_step((1, 3, 3), 256)

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
        x = data[2]
        clearsky = data[1]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        z = tf.concat([x, clearsky], 1)  # Late combining of the metadata.
        x = self.d4(z)
        # x = self.d4(x)
        x = self.d5(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv3d_1 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        conv3d_2 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        conv3d_3 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling3D(pool_size=(1, 2, 2))

        return Sequential([conv3d_1, conv3d_2, conv3d_3, max_pool])

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.ratio = 1
        config.time_interval_min = 60
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

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda dummy_target, metadata, image, target_ghi: (
                dummy_target,
                metadata,
                self.scaling_image.normalize(image),
                target_ghi,
            )
        )


class CNN3D_ClearskyV2(base.Model):
    """Create Conv3D model."""

    # Using the architecture from Tran et al.
    # https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf

    def __init__(self, num_images=4):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.num_images = num_images
        self.conv1a = Conv3D(64, (3, 3, 3), padding="same")
        self.pool1 = MaxPooling3D(pool_size=(1, 2, 2), padding="same")
        self.dropout1 = Dropout(0.1)
        self.conv2a = Conv3D(128, (3, 3, 3), padding="same")
        self.pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.conv3a = Conv3D(256, (3, 3, 3), padding="same")
        self.conv3b = Conv3D(256, (3, 3, 3), padding="same")
        self.pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.conv4a = Conv3D(512, (3, 3, 3), padding="same")
        self.conv4b = Conv3D(512, (3, 3, 3), padding="same")
        self.pool4 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.conv5a = Conv3D(512, (3, 3, 3), padding="same")
        self.conv5b = Conv3D(512, (3, 3, 3), padding="same")
        self.pool5 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.flatten = Flatten()
        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(521, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(4)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = data[2]
        # x = tf.math.l2_normalize(x, axis=0)
        clearsky = data[1]

        x = self.conv1a(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2a(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)  # Here
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        z = tf.concat([x, clearsky], 1)  # Late combining of the metadata.
        x = self.d4(z)
        # x = self.d4(x)
        x = self.d5(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv3d_1 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        conv3d_2 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        conv3d_3 = Conv3D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling3D(pool_size=(1, 2, 2))

        return Sequential([conv3d_1, conv3d_2, conv3d_3, max_pool])

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.ratio = 1
        config.time_interval_min = 60
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

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda dummy_target, metadata, image, target_ghi: (
                dummy_target,
                metadata,
                image,
                target_ghi,
            )
        )
