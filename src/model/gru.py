from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import autoencoder, base

logger = logging.create_logger(__name__)

NAME = "GRU"


class GRU(base.Model):
    """Create GRU Model based on the embeddings created with the encoder."""

    def __init__(self, encoder=None, num_images=6, time_interval_min=30, dropout=0.20):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.num_images = num_images
        self.time_interval_min = time_interval_min

        self.scaling_image = preprocessing.min_max_scaling_images()
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()

        if encoder is None:
            self.encoder = autoencoder.Encoder()
            self.encoder.load(autoencoder.BEST_MODEL_WEIGHTS)
        else:
            self.encoder = encoder

        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(dropout)

        self.gru1 = layers.GRU(512, return_sequences=True)
        self.gru2 = layers.GRU(256)

        self.d1 = layers.Dense(512)
        self.d2 = layers.Dense(256)
        self.d3 = layers.Dense(128)
        self.d4 = layers.Dense(4)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        images, clearsky = data

        x = self.gru1(images)
        x = self.gru2(x)

        x = tf.concat([x, clearsky], 1)

        x = self.d1(x)
        if training:
            x = self.dropout(x)
        x = self.d2(x)
        if training:
            x = self.dropout(x)
        x = self.d3(x)
        if training:
            x = self.dropout(x)
        x = self.d4(x)

        return x

    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = self.time_interval_min
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
        Data is now (features, target).
        """

        def encoder(images):
            return self.encoder(images, training=False)

        def preprocess(images, clearsky, target_ghi):
            images = self.scaling_image.normalize(images)
            clearsky = self.scaling_ghi.normalize(clearsky)
            target_ghi = self.scaling_ghi.normalize(target_ghi)
            # Warp the encoder preprocessing in a py function
            # because its size is not known at compile time.
            features = tf.py_function(func=encoder, inp=[images], Tout=tf.float32)
            return (features, clearsky, target_ghi)

        return dataset.map(preprocess)
