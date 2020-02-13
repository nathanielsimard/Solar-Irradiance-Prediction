import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "Conv2D"


class CNN2D(base.Model):
    """Create Conv2D model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.scaling_target = preprocessing.MinMaxScaling(
            preprocessing.TARGET_GHI_MIN, preprocessing.TARGET_GHI_MAX
        )

        # Check if necessary
        input_shape = (64, 64, 5)
        self.conv1 = Conv2D(
            64, kernel_size=(5, 5), input_shape=input_shape, activation="relu"
        )
        self.mp1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2 = Conv2D(128, kernel_size=(5, 5), activation="relu")
        self.mp2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3 = Conv2D(128, kernel_size=(3, 3), activation="relu")
        self.mp3 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()

        self.d1 = Dense(256, activation="relu")
        self.d2 = Dense(4)

    def call(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def config(self, training=False) -> dataloader.Config:
        config = default_config()
        config.num_images = 1
        config.ratio = 0.01
        config.features = [dataloader.Feature.image, dataloader.Feature.target_ghi]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(
            lambda image, target_ghi: (
                self.scaling_image.normalize(image),
                self._preprocess_target(target_ghi),
            )
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        current_target_only = target_ghi[0:1]
        return self.scaling_target.normalize(current_target_only)
