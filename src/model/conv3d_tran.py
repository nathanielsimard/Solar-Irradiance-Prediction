from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "CNN3DTranClearsky"


class CNN3DTranClearsky(base.Model):
    """Create Conv3D model."""

    # Using the architecture from Tran et al.
    # https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf

    def __init__(self, num_images=4, time_interval_min=60):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.min_max_scaling_images()
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()
        self.num_images = num_images
        self.time_interval_min = time_interval_min
        self.inputdropout = layers.Dropout(0.5)
        self.conv1a = layers.Conv3D(64, (3, 3, 3), padding="same")
        self.pool1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")
        self.dropout1 = layers.Dropout(0.1)
        self.batchnorm1 = layers.BatchNormalization()
        self.conv2a = layers.Conv3D(128, (3, 3, 3), padding="same")
        self.pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout2 = layers.Dropout(0.1)
        self.batchnorm2 = layers.BatchNormalization()
        self.conv3a = layers.Conv3D(256, (3, 3, 3), padding="same")
        self.conv3b = layers.Conv3D(256, (3, 3, 3), padding="same")
        self.pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout3 = layers.Dropout(0.1)
        self.batchnorm3 = layers.BatchNormalization()
        self.conv4a = layers.Conv3D(512, (3, 3, 3), padding="same")
        self.conv4b = layers.Conv3D(512, (3, 3, 3), padding="same")
        self.pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout4 = layers.Dropout(0.1)
        self.batchnorm4 = layers.BatchNormalization()
        self.conv5a = layers.Conv3D(512, (3, 3, 3), padding="same")
        self.conv5b = layers.Conv3D(512, (3, 3, 3), padding="same")
        self.pool5 = layers.MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout5 = layers.Dropout(0.1)
        self.batchnorm5 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(1048, activation="relu")
        self.d2 = layers.Dense(521, activation="relu")
        self.d3 = layers.Dense(256, activation="relu")
        self.d4 = layers.Dense(256, activation="relu")
        self.d5 = layers.Dense(4)

    def call(self, data: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network."""
        meta, images = data
        x = self.conv1a(images)
        x = self.pool1(x)

        if training:
            x = self.dropout1(x, training)

        x = self.batchnorm1(x, training)
        x = self.conv2a(x)
        x = self.pool2(x)

        if training:
            x = self.dropout2(x, training)

        x = self.batchnorm2(x, training)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        if training:
            x = self.dropout3(x, training)

        x = self.batchnorm3(x, training)
        x = self.conv4a(x)
        x = self.conv4b(x)  # Here
        x = self.pool4(x)

        if training:
            x = self.dropout4(x, training)

        x = self.batchnorm4(x, training)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)

        if training:
            x = self.dropout5(x, training)

        x = self.batchnorm5(x, training)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        z = tf.concat([x, meta], 1)  # Late combining of the metadata.
        x = self.d4(z)
        x = self.d5(x)
        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.ratio = 1
        config.time_interval_min = 60
        config.features = [
            dataloader.Feature.metadata,
            dataloader.Feature.image,
            dataloader.Feature.target_ghi,
        ]

        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore

        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""

        def preprocess(metadata, images, target_ghi):
            images = self.scaling_image.normalize(images)
            metadata = self.scaling_ghi.normalize(metadata)
            target_ghi = self.scaling_ghi.normalize(target_ghi)
            return metadata, images, target_ghi

        return dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
