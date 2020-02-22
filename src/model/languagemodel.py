from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "LanguageModel"


class LanguageModel(base.Model):
    """Create Language Model to predict the futur images."""

    def __init__(self, encoder, num_images=6, num_features=16 * 16 * 32):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.num_images = num_images

        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.encoder = encoder
        self.num_features = num_features

        self.flatten = layers.Flatten()

        self.l1 = layers.GRU(num_features, return_sequences=True)
        self.l2 = layers.GRU(num_features, return_sequences=True)
        self.l3 = layers.GRU(num_features, return_sequences=True)

    def call(self, x: Tuple[tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = x[0]
        shape = x.shape

        x = tf.reshape(x, (shape[0], shape[1], self.num_features))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = tf.reshape(x, shape)
        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = 60
        config.skip_missing_past_images = True
        config.features = [dataloader.Feature.image]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def predict_next_images(self, images: Tuple[tf.Tensor], num_images=6):
        """Predict the next images from the original images.

        Args:
            images: Tensor of shape (num_images, width, height, channels)
                Images must not be scaled, crop or anything special.
        """
        images = self._preprocess_images(images)

        for _ in range(num_images):
            predictions = self.call((images))
            images = tf.concat([images[:, 0:1], predictions[:, :]], 1)

        # Remove batch size
        images = images[0]
        # Only keep the last images
        predictions = images[-num_images:]

        return predictions

    def _preprocess_images(self, images: tf.Tensor) -> tf.Tensor:
        count_nonzero = [tf.math.count_nonzero(image) for image in images]

        images_preprocessed = self.scaling_image.normalize(images)
        images_encoded = self.encoder(images_preprocessed, training=False)

        inputs: List[tf.Tensor] = []

        for i, image in enumerate(images_encoded):
            if count_nonzero[i] == 0:
                logger.info(f"Missing Image, {images[i].numpy()}")

                if inputs is None:
                    logger.info(f"First image, skipping")
                    continue

                tensor = tf.constant(inputs)
                # Introduce batch size dim
                tensor = tf.expand_dims(tensor)

                predictions = self.call((tensor))
                # Append prediction instead of missing image.
                inputs.append(predictions[:, -1])
            else:
                inputs.append(image)

        images = tf.constant(inputs)
        # Introduce batch size dim
        return tf.expand_dims(images)

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Encode images and return it as input and target."""

        def encoder(images):
            return self.encoder((images), False)

        def preprocess(images):
            images = self.scaling_image.normalize(images)
            image_features = tf.py_function(func=encoder, inp=[images], Tout=tf.float32)
            return (image_features[0:-1], image_features[1:])

        return dataset.map(preprocess).cache()
