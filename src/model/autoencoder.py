from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, UpSampling2D

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME_AUTOENCODER = "Autoencoder"
NAME_DECODER = "Decoder"
NAME_ENCODER = "Encoder"


class Encoder(base.Model):
    """Create Image Encoder model."""

    def __init__(self, dropout=0.5):
        """Initialize the architecture."""
        super().__init__(NAME_ENCODER)
        self.conv1 = Conv2D(
            64, kernel_size=(3, 3), activation="relu", strides=2, padding="same"
        )
        self.conv2 = Conv2D(
            64, kernel_size=(3, 3), activation="relu", strides=2, padding="same"
        )
        self.conv3 = Conv2D(
            32, kernel_size=(3, 3), activation="relu", strides=2, padding="same"
        )
        self.dropout = Dropout(dropout)

    def call(self, x: tf.Tensor, training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)

        if training:
            x = self.dropout(x)

        x = self.conv2(x)

        if training:
            x = self.dropout(x)

        x = self.conv3(x)

        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        raise Exception("Config should be passe to the model using the encoder.")


class Decoder(base.Model):
    """Create Image Decoder model."""

    def __init__(self, num_channels, dropout=0.5):
        """Initialize a decoder with a fixed number of channels."""
        super().__init__(NAME_DECODER)
        self.conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv3 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv4 = Conv2D(num_channels, kernel_size=(3, 3), padding="same")

        self.up_sampling = UpSampling2D((2, 2))
        self.dropout = Dropout(dropout)

    def call(self, x: tf.Tensor, training=False):
        """Decode a compressed image into the original image."""
        x = self.conv1(x)
        x = self.up_sampling(x)

        if training:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.up_sampling(x)

        if training:
            x = self.dropout(x)

        x = self.conv3(x)
        x = self.up_sampling(x)

        if training:
            x = self.dropout(x)

        x = self.conv4(x)

        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        raise Exception("Config should be passe to the model using the decoder.")


class Autoencoder(base.Model):
    """Create Image Auto-Encoder model."""

    def __init__(self, dropout=0.3):
        """Initialize the autoencoder."""
        super().__init__(NAME_AUTOENCODER)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        self.default_config = default_config()
        self.default_config.num_images = 1
        self.default_config.features = [
            dataloader.Feature.image,
            dataloader.Feature.target_ghi,
        ]

        num_channels = len(self.default_config.channels)

        self.encoder = Encoder(dropout=dropout)
        self.decoder = Decoder(num_channels, dropout=dropout)

    def call(self, data: Tuple[tf.Tensor], training=False):
        """Encode than decode the image."""
        x = data[0]

        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)

        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = self.default_config

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the image to return two times the same image."""
        return dataset.map(
            lambda image, _: self._preprocess(image),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def save(self, instance: str):
        """Override the save method to save the encoder and decoder."""
        self.encoder.save(instance)
        self.decoder.save(instance)

    def load(self, instance: str):
        """Override the load method to load the encoder and decoder."""
        self.encoder.load(instance)
        self.decoder.load(instance)

    def _preprocess(self, image: tf.Tensor) -> tf.Tensor:
        scaled_image = self.scaling_image.normalize(image)
        return (scaled_image, scaled_image)
