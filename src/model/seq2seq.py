from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import autoencoder, base

logger = logging.create_logger(__name__)

NAME = "Seq2Seq"


class Seq2Seq(base.Model):
    """Create Language Model to predict the futur images."""

    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = self.time_interval_min
        config.skip_missing_past_images = True
        config.features = [dataloader.Feature.image]

        return config

    def predict_next_images(self, images: tf.Tensor, num_images=6):
        """Predict the next images from the original images.

        Args:
            images: Tensor of shape (num_images, width, height, channels)
                Images must not be scaled, crop or anything special.
            num_images: Number of futur images to generate.
        """
        images = self._preprocess_images(images)

        for _ in range(num_images):
            predictions = self.call((images,))
            images = tf.concat([images[:, :], predictions[:, :-1]], 1)

        # Remove batch size
        images = images[0]
        # Only keep the last images
        predictions = images[-num_images:]

        return predictions

    def _preprocess_images(self, images: tf.Tensor) -> tf.Tensor:
        count_nonzero = [tf.math.count_nonzero(image) for image in images]

        images_preprocessed = self.scaling_image.normalize(images)
        images_encoded = self.encoder(images_preprocessed, training=False)
        images_encoded = tf.expand_dims(images_encoded, 0)

        for i in range(images_encoded.shape[1]):
            if count_nonzero[i] == 0:
                logger.debug(f"Missing Image, {i}")

                if i == 0:
                    logger.debug("First image, skipping")
                    continue

                predictions = self.call((images_encoded[:, :i],))
                logger.debug(f"Replacing image {i} with prediction")
                images_encoded = tf.concat(
                    [
                        images_encoded[:, :i],  # Previous
                        predictions[:, -1:],  # Predictions
                        images_encoded[:, i + 1 :],  # Next
                    ],
                    1,
                )

        return images_encoded

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Encode images and return it as input and target."""

        def encoder(images):
            return self.encoder(images, training=False)

        def preprocess(images):
            images = self.scaling_image.normalize(images)
            image_features = tf.py_function(func=encoder, inp=[images], Tout=tf.float32)
            return (image_features[0:-1], image_features[1:])

        return dataset.map(preprocess)


class ConvLSTM(Seq2Seq):
    """Use ConvLSTM2D as recurent layers."""

    def __init__(
        self, encoder=None, num_images=6, time_interval_min=60, num_channels=32
    ):
        """Initialize the architecture."""
        super().__init__(NAME + "ConvLSTM")
        self.num_images = num_images
        self.time_interval_min = time_interval_min

        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        if encoder is None:
            self.encoder = autoencoder.Encoder()
            self.encoder.load(autoencoder.BEST_MODEL_WEIGHTS)
        else:
            self.encoder = encoder

        self.l1 = layers.ConvLSTM2D(
            64, kernel_size=(3, 3), padding="same", return_sequences=True
        )
        self.l2 = layers.ConvLSTM2D(
            64, kernel_size=(3, 3), padding="same", return_sequences=True
        )
        self.l3 = layers.ConvLSTM2D(
            num_channels, kernel_size=(3, 3), padding="same", return_sequences=True
        )

    def call(self, x: Tuple[tf.Tensor], training=False):
        """Performs the forward pass in the neural network."""
        x = x[0]

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class Gru(Seq2Seq):
    """Use GRU as recurent layers."""

    def __init__(
        self,
        encoder=None,
        num_images=6,
        time_interval_min=60,
        num_features=16 * 16 * 32,
    ):
        """Initialize the architecture."""
        super().__init__(NAME + "GRU")
        self.num_images = num_images
        self.time_interval_min = time_interval_min
        self.num_features = num_features

        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        if encoder is None:
            self.encoder = autoencoder.Encoder()
            self.encoder.load(autoencoder.BEST_MODEL_WEIGHTS)
        else:
            self.encoder = encoder

        self.l1 = layers.GRU(1024, return_sequences=True)
        self.l2 = layers.GRU(1024, return_sequences=True)
        self.l3 = layers.GRU(1024, return_sequences=True)
        self.l4 = layers.Dense(num_features)

    def call(self, x: Tuple[tf.Tensor], training=False):
        """Performs the forward pass in the neural network."""
        x = x[0]
        shape = x.shape  # type: ignore
        x = tf.reshape(x, (shape[0], shape[1], self.num_features))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = tf.reshape(x, shape)
        return x
