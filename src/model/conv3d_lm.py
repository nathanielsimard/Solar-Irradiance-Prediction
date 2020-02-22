from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base, languagemodel

logger = logging.create_logger(__name__)

NAME = "Conv3D-LanguageModel"


class Conv3D(base.Model):
    """Create Conv3D model."""

    def __init__(self, language_model: languagemodel.LanguageModel):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.language_model = language_model

        self.scaling_ghi = preprocessing.MinMaxScaling(
            preprocessing.TARGET_GHI_MIN, preprocessing.TARGET_GHI_MIN
        )

        self.flatten = layers.Flatten()
        self.max_pool = layers.MaxPooling3D((1, 2, 2))

        self.conv1 = layers.Conv3D(
            64, kernel_size=(1, 3, 3), padding="same", activation="relu"
        )
        self.conv2 = layers.Conv3D(
            128, kernel_size=(1, 3, 3), padding="same", activation="relu"
        )
        self.conv3 = layers.Conv3D(
            128, kernel_size=(1, 3, 3), padding="same", activation="relu"
        )

        self.d1 = layers.Dense(512, activation="relu")
        self.d2 = layers.Dense(256, activation="relu")
        self.d3 = layers.Dense(4)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        images, clearsky = data[0]

        x = self.conv1(images)
        x = self.conv2(x)

        x = self.max_pool(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = tf.concat([x, clearsky], 1)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        if not training:
            return self.scaling_ghi.original(x)

        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.language_model.num_images
        config.time_interval_min = self.language_model.time_interval_min
        config.features = [
            dataloader.Feature.image,
            dataloader.Feature.metadata,
            dataloader.Feature.target_ghi,
        ]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""

        def generate(images):
            print(images.shape)
            futur_images = self.language_model.predict_next_images(images, num_images=6)
            print(futur_images.shape)
            # Return images at t0, t1, t3, t6
            im = tf.concat(
                [futur_images[0], futur_images[1], futur_images[2], futur_images[5]], 0
            )
            print(im.shape)
            return im

        def preprocess(images, target_csm, target_ghi):
            target_csm = self.scaling_ghi.normalize(target_csm)
            target_ghi = self.scaling_ghi.normalize(target_ghi)
            # Warp the encoder preprocessing in a py function
            # because its size is not known at compile time.
            images = tf.py_function(func=generate, inp=[images], Tout=tf.float32)
            return (images, target_csm, target_ghi)

        return dataset.map(preprocess)
