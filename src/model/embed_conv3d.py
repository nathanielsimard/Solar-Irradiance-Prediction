from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import autoencoder, base

logger = logging.create_logger(__name__)

NAME = "EmbedConv3D"
encoder32 = autoencoder.Encoder()
encoder32.load(autoencoder.BEST_MODEL_WEIGHTS)


class Conv3D(base.Model):
    """Create Conv3D Model based on the embeddings created with the Encoder."""

    def __init__(
        self,
        encoder=None,
        num_images=6,
        time_interval_min=30,
        dropout=0.25,
        crop_size=64,
    ):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.num_images = num_images
        self.time_interval_min = time_interval_min
        self.crop_size = crop_size
        self.scaling_image = preprocessing.min_max_scaling_images()
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()

        if encoder is None:
            self.encoder = autoencoder.Encoder()
            self.encoder.load(autoencoder.BEST_MODEL_WEIGHTS)
        else:
            self.encoder = encoder

        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(dropout)

        self.conv1 = layers.Conv3D(
            32, kernel_size=(3, 3, 3), padding="same", activation="relu"
        )
        self.conv2 = layers.Conv3D(
            32, kernel_size=(3, 3, 3), padding="same", activation="relu"
        )
        self.conv3 = layers.Conv3D(
            32, kernel_size=(3, 3, 3), padding="same", activation="relu"
        )
        self.conv4 = layers.Conv3D(
            32, kernel_size=(3, 3, 3), padding="same", activation="relu"
        )

        self.max_pool = layers.MaxPooling3D((2, 2, 2))

        self.d1 = layers.Dense(256)
        self.d2 = layers.Dense(128)
        self.d3 = layers.Dense(4)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        images, target_csm = data

        x = self.conv1(images)
        x = self.conv2(x)

        if training:
            x = self.dropout(x)

        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.conv4(x)

        if training:
            x = self.dropout(x)

        x = self.flatten(x)

        x = tf.concat([x, target_csm], 1)

        x = self.d1(x)
        if training:
            x = self.dropout(x)
        x = self.d2(x)
        if training:
            x = self.dropout(x)
        x = self.d3(x)

        return x

    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = self.time_interval_min
        config.crop_size = (self.crop_size, self.crop_size)
        config.features = [
            dataloader.Feature.image,
            dataloader.Feature.metadata,
            dataloader.Feature.target_ghi,
        ]

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply preprocessing specifitly for this model.

        Extract the features from the image with the encoder.
        """

        def encoder(images):
            return self.encoder((images), training=False)

        def crop_image(image: tf.Tensor):
            """Performs dynamic cropping of an image."""
            image_size_x = image.shape[0]
            image_size_y = image.shape[1]
            pixel = self.crop_size
            start_x = image_size_x // 2 - pixel // 2
            end_x = image_size_x // 2 + pixel // 2
            start_y = image_size_y // 2 - pixel // 2
            end_y = image_size_y // 2 + pixel // 2
            cropped_image = image[start_x:end_x, start_y:end_y, :]
            return cropped_image

        def preprocess(images, target_csm, target_ghi):
            images = self.scaling_image.normalize(images)
            target_csm = self.scaling_ghi.normalize(target_csm)
            target_ghi = self.scaling_ghi.normalize(target_ghi)
            # Warp the encoder preprocessing in a py function
            # because its size is not known at compile time.
            images = tf.py_function(func=crop_image, inp=[images], Tout=tf.float32)
            images = tf.py_function(func=encoder, inp=[images], Tout=tf.float32)
            return (images, target_csm, target_ghi)

        return dataset.map(preprocess)
