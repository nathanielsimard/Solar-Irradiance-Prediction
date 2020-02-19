import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "Clearsky1"


class Clearsky(base.Model):
    """Create Conv2D model."""

    def __init__(self, encoder):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.encoder = encoder

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(1)

    def call(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.features = [
            dataloader.Feature.image,
            dataloader.Feature.target_csm,
            dataloader.Feature.target_ghi,
        ]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""

        def preprocess(image, clearsky, target_ghi):
            image = self.scaling_image.normalize(image)
            image_encoded = self.encoder(image)
            image_features = self.flatten(image_encoded)

            clearsky = self._preprocess_target(clearsky)
            target_ghi = self._preprocess_target(target_ghi)

            return (tf.concat([image_features, clearsky]), target_ghi)

        return dataset.map(
            preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]
