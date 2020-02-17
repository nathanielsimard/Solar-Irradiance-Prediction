import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
    LSTM,
    Conv2D,
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

    def __init__(self, num_images=8):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.num_images = num_images

        self.conv1 = TimeDistributed(Conv2D(64, (5,5), activation="relu"), input_shape=(self.num_images, 64,64,5))
        self.mp1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))
        self.flat = TimeDistributed(Flatten())
        self.lstm = LSTM(units=1024, return_sequences=False)

        self.d1 = Dense(1024, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(4)

    def call(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.mp1(x)
        print(x.shape)
        x = self.flat(x)
        print(x.shape)

        x = self.lstm(x)
        print(x.shape)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x

    def _cnn(self) -> Sequential:
        conv1 = self._convolution_step((5, 5), 32)
        conv2 = self._convolution_step((3, 3), 64)
        conv3 = self._convolution_step((3, 3), 64)

        return Sequential([conv1, conv2, conv3])

    def _convolution_step(self, kernel_size, channels):
        conv3d_1 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv3d_2 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling2D(pool_size=(2, 2))

        return Sequential([conv3d_1, conv3d_2, max_pool])

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.ratio = 0.01
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
