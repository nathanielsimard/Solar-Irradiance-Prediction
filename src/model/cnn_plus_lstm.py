import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
    LSTM,
    Conv2D,
    Dropout,
)
from tensorflow.keras.models import Sequential

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "CNN_LSTM"


class CNNLSTM(base.Model):
    """Create ConvLSTM model."""

    def __init__(self, num_images=16):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.num_images = num_images

        self.conv1 = self._convolution_step((3, 3), 64, first=True)
        self.drop1 = Dropout(0.2)
        self.conv2 = self._convolution_step((3, 3), 128)
        self.drop2 = Dropout(0.2)
        self.conv3 = self._convolution_step((3, 3), 256)
        self.drop3 = Dropout(0.2)
        self.flat = TimeDistributed(Flatten())
        self.d1 = TimeDistributed(Dense(512))
        self.drop4 = Dropout(0.3)
        self.lstm = LSTM(units=16, return_sequences=False)

        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(4)

    def call(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)
        if training:
            x = self.drop1(x)
        x = self.conv2(x)
        if training:
            x = self.drop2(x)
        x = self.conv3(x)
        if training:
            x = self.drop3(x)
        x = self.flat(x)
        x = self.d1(x)
        if training:
            x = self.drop4(x)
        x = self.lstm(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x

    def _convolution_step(self, kernel_size, channels, first=False):
        conv2 = TimeDistributed(Conv2D(channels, kernel_size, activation="relu"))
        max_pool = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))

        if first:
            conv1 = TimeDistributed(
                Conv2D(channels, kernel_size, activation="relu"),
                input_shape=(self.num_images, 64, 64, 5),
            )
        else:
            conv1 = TimeDistributed(Conv2D(channels, kernel_size, activation="relu"))

        return Sequential([conv1, conv2, max_pool])

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.ratio = 0.15
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
