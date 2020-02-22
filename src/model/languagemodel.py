import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, GRU

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "LanguageModel"


class LanguageModel(base.Model):
    """Create Language Model to predict the futur images."""

    def __init__(self, encoder, num_images=6, num_features=2048):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.num_images = num_images

        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.encoder = encoder
        self.num_features = num_features

        self.l1 = ConvLSTM2D(16, kernel_size=(3,3), padding="same", return_sequences=True)
        self.l2 = ConvLSTM2D(16, kernel_size=(3,3), padding="same", return_sequences=True)
        self.l3 = ConvLSTM2D(5, kernel_size=(3,3), padding="same", return_sequences=True)

        self.flatten = Flatten()

        # self.l1 = GRU(num_features, return_sequences=True,)
        # self.l2 = GRU(num_features, return_sequences=True,)
        # self.l3 = GRU(num_features, return_sequences=True,)

    def call(self, x, training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = x[0]
        #shape = x.shape

        #x = tf.reshape(x, (shape[0], shape[1], self.num_features))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        #x = tf.reshape(x, shape)
        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = 60
        config.features = [dataloader.Feature.image]

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
            return self.encoder((images), False)

        def preprocess(images):
            images = self.scaling_image.normalize(images)
            # image_features = tf.py_function(func=encoder, inp=[images], Tout=tf.float32)
            return (images[0:-1], images[1:])

        return dataset.map(preprocess).cache()
