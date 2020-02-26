import tensorflow as tf
from typing import Tuple
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
    LSTM,
    Conv2D,
    Dropout,
    LeakyReLU,
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

    def __init__(self, num_images=4, num_outputs=4):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        self.scaling_target = preprocessing.MinMaxScaling(
            preprocessing.TARGET_GHI_MIN, preprocessing.TARGET_GHI_MAX
        )
        self.num_images = num_images
        self.num_outputs = num_outputs

        self.conv1 = self._convolution_step((3, 3), 16, first=True)
        self.drop1 = Dropout(0.2)
        self.conv2 = self._convolution_step((3, 3), 32)
        self.drop2 = Dropout(0.2)
        self.flat = TimeDistributed(Flatten())
        self.d1 = TimeDistributed(Dense(256))
        self.drop4 = Dropout(0.3)

        self.lstm = LSTM(units=4)

        self.d4 = Dense(self.num_outputs)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        images, clearsky = data
        x = self.conv1(images)
        if training:
            x = self.drop1(x)
        x = self.conv2(x)
        if training:
            x = self.drop2(x)
        x = self.flat(x)
        x = self.d1(x)
        if training:
            x = self.drop4(x)

        x = self.lstm(x)
        x = tf.concat([x, clearsky], axis=1)
        x = self.d4(x)

        return x

    def _convolution_step(self, kernel_size, channels, first=False):
        conv2 = TimeDistributed(Conv2D(channels, kernel_size))
        act2 = TimeDistributed(LeakyReLU(0.2))
        max_pool = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))

        if first:
            conv1 = TimeDistributed(
                Conv2D(channels, kernel_size), input_shape=(self.num_images, 64, 64, 5),
            )
            act1 = TimeDistributed(LeakyReLU(0.2))
        else:
            conv1 = TimeDistributed(Conv2D(channels, kernel_size))
            act1 = TimeDistributed(LeakyReLU(0.2))

        return Sequential([conv1, act1, conv2, act2, max_pool])

    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = 30
        config.features = [
            dataloader.Feature.image,
            dataloader.Feature.metadata,
            dataloader.Feature.target_ghi,
        ]

        return config

    def preprocess(self, dataset: tf.data.Dataset, training=True) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""

        def preprocess(images, target_csm, target_ghi):
            images = self.scaling_image.normalize(images)
            target_csm = self.scaling_target.normalize(target_csm)
            target_ghi = self.scaling_target.normalize(target_ghi)

            return images, target_csm, target_ghi

        return dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
