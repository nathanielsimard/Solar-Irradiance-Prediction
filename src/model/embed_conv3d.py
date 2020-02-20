import tensorflow as tf
from tensorflow.keras import layers

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "EmbedConv3D"


class Conv3D(base.Model):
    """Create Language Model to predict the futur images."""

    def __init__(self, encoder, num_images=6, time_interval_min=30, dropout=0.20):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.num_images = num_images
        self.time_interval_min = time_interval_min

        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        self.encoder = encoder
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(dropout)

        self.conv1 = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")
        self.conv2 = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")
        self.conv3 = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")
        self.conv4 = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")

        self.max_pool = layers.MaxPooling3D((2, 2, 2))

        self.d1 = layers.Dense(256)
        self.d2 = layers.Dense(128)
        self.d3 = layers.Dense(4)

    def call(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        batch_size = x.shape[0]
        num_clearsky = 4

        image = x[:, :, 0:-num_clearsky]
        image = tf.reshape(image, (batch_size, self.num_images, 8, 8, 32))
        clearsky = x[:, :, -num_clearsky:]

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        # Add only latest clearsky
        x = tf.concat([x, clearsky[:, -1, :]], 1)

        x = self.d1(x)
        if training:
            x = self.dropout(x)
        x = self.d2(x)
        if training:
            x = self.dropout(x)
        x = self.d3(x)

        return x

    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = self.num_images
        config.time_interval_min = self.time_interval_min
        config.features = [
            dataloader.Feature.image,
            dataloader.Feature.clearsky,
            dataloader.Feature.target_ghi,
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

        def encoder(images, clearsky):
            images_encoded = self.encoder((images), False)
            image_features = self.flatten(images_encoded)
            features = tf.concat([image_features, clearsky], 1)
            return features

        def preprocess(images, clearsky, target_ghi):
            images = self.scaling_image.normalize(images)
            # Warp the encoder preprocessing in a py function
            # because its size is not known at compile time.
            features = tf.py_function(
                func=encoder, inp=[images, clearsky], Tout=tf.float32
            )
            # Every image feature also has the 4 clearsky predictions.
            return (features, target_ghi)

        return dataset.map(preprocess)
