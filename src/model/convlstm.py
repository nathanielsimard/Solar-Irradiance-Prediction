import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling3D,
    ConvLSTM2D,
    TimeDistributed,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "ConvLSTM"


class CONVLSTM(base.Model):
    """Create ConvLSTM model."""

    def __init__(self, num_images=8):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.num_images = num_images

        self.conv_lstm1 = self._convlstm_block((5, 5), 32)
        self.conv_lstm2 = self._convlstm_block((3, 3), 64)
        self.conv_lstm3 = self._convlstm_block((3, 3), 64)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(4)

    def call(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv_lstm1(x)
        x = self.conv_lstm2(x)
        x = self.conv_lstm3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x

    def _convlstm_block(self, kernel_size, channels):
        convlstm_1 = ConvLSTM2D(channels, kernel_size=kernel_size, activation="tanh", return_sequences=True)
        convlstm_2 = ConvLSTM2D(channels, kernel_size=kernel_size, activation="tanh", return_sequences=True)
        max_pool = MaxPooling3D(pool_size=(1, 2, 2))

        return Sequential([convlstm_1, convlstm_2, max_pool])

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.features = [dataloader.Feature.image, dataloader.Feature.target_ghi]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda image, target_ghi: (
                self.scaling_image.normalize(image),
                target_ghi,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
