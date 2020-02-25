
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, Flatten, MaxPooling3D
from tensorflow.keras.models import Sequential

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
    def __init__(self, num_images=4, time_interval_min):
        """Initialize the architecture."""
        super().__init__(NAME_CLEARSKY_V2)
        self.scaling_image = preprocessing.min_max_scaling_images()
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()
        self.num_images = num_images
        self.time_interval_min = time_interval_min
        self.inputdropout = Dropout(0.5)
        self.conv1a = Conv3D(64, (3, 3, 3), padding="same")
        self.pool1 = MaxPooling3D(pool_size=(1, 2, 2), padding="same")
        self.dropout1 = Dropout(0.1)
        self.batchnorm1 = BatchNormalization()
        self.conv2a = Conv3D(128, (3, 3, 3), padding="same")
        self.pool2 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout2 = Dropout(0.1)
        self.batchnorm2 = BatchNormalization()
        self.conv3a = Conv3D(256, (3, 3, 3), padding="same")
        self.conv3b = Conv3D(256, (3, 3, 3), padding="same")
        self.pool3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout3 = Dropout(0.1)
        self.batchnorm3 = BatchNormalization()
        self.conv4a = Conv3D(512, (3, 3, 3), padding="same")
        self.conv4b = Conv3D(512, (3, 3, 3), padding="same")
        self.pool4 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout4 = Dropout(0.1)
        self.batchnorm4 = BatchNormalization()
        self.conv5a = Conv3D(512, (3, 3, 3), padding="same")
        self.conv5b = Conv3D(512, (3, 3, 3), padding="same")
        self.pool5 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")
        self.dropout5 = Dropout(0.1)
        self.batchnorm5 = BatchNormalization()
        self.flatten = Flatten()
        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(521, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(4)
    def call(self, data: Tuple[tf.Tensor, tf.Tensor], training=False):
        """Performs the forward pass in the neural network.
        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        meta, images = data
        x = self.conv1a(images)
        x = self.pool1(x)
        x = self.dropout1(x, training)
        x = self.batchnorm1(x, training)
        x = self.conv2a(x)
        x = self.pool2(x)
        x = self.dropout2(x, training)
        x = self.batchnorm2(x, training)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        x = self.dropout3(x, training)
        x = self.batchnorm3(x, training)
        x = self.conv4a(x)
        x = self.conv4b(x)  # Here
        x = self.pool4(x)
        x = self.dropout4(x, training)
        x = self.batchnorm4(x, training)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)
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
    def config(self) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = self.time_interval_min
        config.features = [
            dataloader.Feature.metadata,
            dataloader.Feature.image,
            dataloader.Feature.target_ghi,
        ]
        return config
    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        def preprocess(metadata, images, target_ghi):
            images = self.scaling_image.normalize(images)
            metadata = self.scaling_ghi.normalize(metadata)
            target_ghi = self.scaling_ghi.normalize(target_ghi)
            return metadata, images, target_ghi
        return dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

