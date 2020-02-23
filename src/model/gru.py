from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "GRU"


class GRU(base.Model):
    """Create GRU Model based on the embeddings created with the encoder."""

    def __init__(self, encoder, num_images=6, time_interval_min=30, dropout=0.20):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.num_images = num_images
        self.time_interval_min = time_interval_min

        self.scaling_image = preprocessing.min_max_scaling_images()
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()

        self.encoder = encoder
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(dropout)

        self.gru1 = layers.GRU(512)

        self.d1 = layers.Dense(512)
        self.d2 = layers.Dense(256)
        self.d3 = layers.Dense(128)
        self.d4 = layers.Dense(4)

    def call(self, data: Tuple[tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = data[0]
        x = self.gru1(x)

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

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = self.time_interval_min
        config.features = [
            dataloader.Feature.image,
            dataloader.Feature.clearsky,
            dataloader.Feature.target_ghi,
        ]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply preprocessing specifitly for this model.

        Extract the features from the image with the encoder.
        Flatten and concatenate them with the clearsky.
        Data is now (features, target).
        """

        def encoder(images, clearsky):
            images_encoded = self.encoder((images), False)
            image_features = self.flatten(images_encoded)
            features = tf.concat([image_features, clearsky], 1)
            return features

        def preprocess(images, clearsky, target_ghi):
            images = self.scaling_image.normalize(images)
            clearsky = self.scaling_ghi.normalize(clearsky)
            target_ghi = self.scaling_ghi.normalize(target_ghi)
            # Warp the encoder preprocessing in a py function
            # because its size is not known at compile time.
            features = tf.py_function(
                func=encoder, inp=[images, clearsky], Tout=tf.float32
            )
            # Every image feature also has the 4 clearsky predictions.
            return (features, target_ghi)

        return dataset.map(preprocess)
