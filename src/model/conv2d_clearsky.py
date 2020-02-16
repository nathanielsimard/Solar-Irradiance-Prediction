import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "CNN2DClearsky"


class CNN2DClearsky(base.Model):
    """Create Conv2D model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        self.conv1 = self._convolution_step((5, 5), 64)
        self.conv2 = self._convolution_step((3, 3), 128)
        self.conv3 = self._convolution_step((3, 3), 256)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(4)

    def call(self, x, meta, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        # z = tf.concat([x, meta], 1)  # Late combining of the metadata.
        #x = self.d4(z)
        x = self.d4(x)
        x = self.d5(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv2d_1 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_2 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_3 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling2D(pool_size=(2, 2))

        return Sequential([conv2d_1, conv2d_2, conv2d_3, max_pool])

    def config(self, training=False, dry_run=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 1
        config.features = [dataloader.Feature.target_ghi, dataloader.Feature.metadata,
                           dataloader.Feature.image, dataloader.Feature.target_cloudiness]
        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore
        if dry_run:
            config.error_strategy = dataloader.ErrorStrategy.stop
        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda target_ghi, metadata, image: (
                target_ghi,
                metadata,
                self.scaling_image.normalize(image),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]


class CNN2DClearskyCloudiness(base.Model):
    """Create Conv2D for target cloudiness model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        self.conv1 = self._convolution_step((5, 5), 64)
        self.conv2 = self._convolution_step((3, 3), 128)
        self.conv3 = self._convolution_step((3, 3), 256)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(5)

    def call(self, x, meta, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        # z = tf.concat([x, meta], 1)  # Late combining of the metadata.
        #x = self.d4(z)
        x = self.d4(x)
        x = self.d5(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv2d_1 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_2 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_3 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling2D(pool_size=(2, 2))

        return Sequential([conv2d_1, conv2d_2, conv2d_3, max_pool])

    def config(self, training=False, dry_run=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 1
        config.features = [dataloader.Feature.target_ghi, dataloader.Feature.metadata,
                           dataloader.Feature.image, dataloader.Feature.target_cloudiness]
        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore
        if dry_run:
            config.error_strategy = dataloader.ErrorStrategy.stop
        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda target_ghi, metadata, image, target_cloudiness: (
                target_ghi,
                metadata,
                self.scaling_image.normalize(image),
                target_cloudiness
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]
