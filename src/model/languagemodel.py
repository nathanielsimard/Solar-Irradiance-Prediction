import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Flatten

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "LanguageModel"


class LanguageModel(base.Model):
    """Create Language Model to predict the futur images."""

    def __init__(self, encoder, num_images=6):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.num_images = num_images

        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.encoder = encoder

        self.flatten = Flatten()
        self.gru = GRU(
            512,
            return_sequences=True,
            stateful=True,
            recurrent_initializer="glorot_uniform",
        )
        self.d1 = Dense(1)

    def call(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.gru(x)
        x = self.d1(x)

        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 0.1
        config.features = [
            dataloader.Feature.image,
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
        Change target to only consider present time.
        Data is now (features, target).
        """

        def encoder(images):
            image_encoded = self.encoder((images), False)
            return self.flatten(image_encoded)

        def preprocess(image):
            image_features = tf.py_function(func=encoder, inp=[image], Tout=tf.float32)
            return (image_features[0:-2], image_features[1:])

        return dataset.map(preprocess).cache()
