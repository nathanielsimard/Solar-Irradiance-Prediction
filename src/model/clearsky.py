from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import autoencoder, base

logger = logging.create_logger(__name__)

NAME = "Clearsky"
NAME_MLP = "ClearskyMLP"


class Clearsky(base.Model):
    """Create Clearsky model.

    This model can not be trained.
    It only predict the clearsky values.
    """

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()

    def call(self, data: Tuple[tf.Tensor], training=False):
        """Return the clearsky values."""
        return data[0]

    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.features = [
            dataloader.Feature.metadata,
            dataloader.Feature.target_ghi,
        ]

        return config

    def load(self, instance):
        """There is nothing to load."""
        pass

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Normalize the ghi."""

        def preprocess(clearsky, target_ghi):
            clearsky = self.scaling_ghi.normalize(clearsky)
            target_ghi = self.scaling_ghi.normalize(target_ghi)

            return (clearsky, target_ghi)

        return dataset.map(preprocess).cache()


class ClearskyMLP(base.Model):
    """Create Clearsky model.

    It only predits the current ghi value based on the current image
    and the current clearsky predictions using the image encoder.
    """

    def __init__(self, encoder: autoencoder.Encoder = None):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.min_max_scaling_images()
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()

        if encoder is None:
            self.encoder = autoencoder.Encoder()
            self.encoder.load(autoencoder.BEST_MODEL_WEIGHTS)
        else:
            self.encoder = encoder

        self.flatten = Flatten()

        self.d1 = Dense(512, activation="relu")
        self.d2 = Dense(128, activation="relu")
        self.d3 = Dense(1)

    def call(self, data: Tuple[tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = data[0]

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        return x

    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.features = [
            dataloader.Feature.image,
            dataloader.Feature.metadata,
            dataloader.Feature.target_ghi,
        ]

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply preprocessing specifitly for this model.

        Extract the features from the image with the encoder.
        Flatten and concatenate them with the clearsky.
        Change target to only consider present time.
        Data is now (features, target).
        """

        def encoder(image):
            # Create Fake Batch Size
            image = tf.expand_dims(image, 0)
            image_encoded = self.encoder((image), training=False)
            # Remove Fake Batch Size
            return self.flatten(image_encoded)[0, :]

        def preprocess(image, clearsky, target_ghi):
            # Normalize inputs
            image = self.scaling_image.normalize(image)
            clearsky = self.scaling_ghi.normalize(clearsky)
            target_ghi = self.scaling_ghi.normalize(target_ghi)

            image_features = tf.py_function(func=encoder, inp=[image], Tout=tf.float32)
            clearsky = self._preprocess_target(clearsky)
            target_ghi = self._preprocess_target(target_ghi)

            features = tf.concat([image_features, clearsky], 0)
            return (features, target_ghi)

        return dataset.map(preprocess).cache()

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]
