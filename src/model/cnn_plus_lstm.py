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
    PReLU,
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

        self.conv1 = self._convolution_step((3, 3), 64, first=True)
        self.drop1 = Dropout(0.2)
        self.conv2 = self._convolution_step((3, 3), 128)
        self.drop2 = Dropout(0.2)
        self.conv3 = self._convolution_step((3, 3), 256)
        self.drop3 = Dropout(0.2)
        self.flat = TimeDistributed(Flatten())
        self.d1 = TimeDistributed(Dense(512))
        self.drop4 = Dropout(0.3)

        self.lstm = LSTM(units=512, return_sequences=False, return_state=False)

        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(128, activation="relu")
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
        x = self.conv3(x)
        if training:
            x = self.drop3(x)
        x = self.flat(x)
        x = self.d1(x)
        if training:
            x = self.drop4(x)

        outputs = self.lstm(x)
        x = tf.concat([outputs, clearsky], axis=1)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x

    def _convolution_step(self, kernel_size, channels, first=False):
        conv2 = TimeDistributed(Conv2D(channels, kernel_size))
        act2 = PReLU()
        max_pool = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))

        if first:
            conv1 = TimeDistributed(
                Conv2D(channels, kernel_size), input_shape=(self.num_images, 64, 64, 5),
            )
            act1 = TimeDistributed(PReLU())
        else:
            conv1 = TimeDistributed(Conv2D(channels, kernel_size))
            act1 = TimeDistributed(PReLU())

        return Sequential([conv1, act1, conv2, act2, max_pool])

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = 60
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

    def preprocess(self, dataset: tf.data.Dataset, training=True) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""

        def preprocess(images, target_csm, target_ghi):
            images = self.scaling_image.normalize(images)
            target_csm = self.scaling_target.normalize(target_csm)
            target_ghi = self.scaling_target.normalize(target_ghi)

            return images, target_csm, target_ghi

        return dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
