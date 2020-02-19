from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import functools

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

from src import logging
from src.data import dataloader, preprocessing
from src.data.train import default_config
from src.model import base

logger = logging.create_logger(__name__)

NAME = "CNN2DClearsky"

ALL_FEATURES = [dataloader.Feature.target_ghi, dataloader.Feature.metadata,
                dataloader.Feature.image, dataloader.Feature.target_cloudiness,
                dataloader.Feature.datetime, dataloader.Feature.location]


class CNN2DClearsky(base.Model):
    """Create Conv2D model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        self.conv1 = self._convolution_step((5, 5), 128)
        self.conv2 = self._convolution_step((3, 3), 256)
        self.dropout2 = Dropout(0.1)
        self.conv3 = self._convolution_step((3, 3), 256)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(4)

    def call(self, x, meta, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)
        x = self.dropout2(self.conv2(x), training)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        z = tf.concat([x, meta], 1)  # Late combining of the metadata.
        x = self.d4(z)
        x = self.d4(x)
        x = self.d5(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv2d_1 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_2 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_3 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling2D(pool_size=(2, 2))

        return Sequential([conv2d_1, conv2d_2, conv2d_3, max_pool])

    def config(self, training=False, dry_run=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 1
        config.features = [ALL_FEATURES]
        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore
        if dry_run:
            config.error_strategy = dataloader.ErrorStrategy.stop
        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda target_ghi, metadata, image: (
                target_ghi,
                metadata,
                self.scaling_image.normalize(image),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]


class CNN2DClearskyCloudiness(base.Model):
    """Create Conv2D for target cloudiness model."""

    def __init__(self):
        """Initialize the architecture."""
        super().__init__(NAME)
        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )

        self.conv1 = self._convolution_step((5, 5), 64)
        self.conv2 = self._convolution_step((3, 3), 128)
        self.conv3 = self._convolution_step((3, 3), 256)

        self.flatten = Flatten()

        self.d1 = Dense(1048, activation="relu")
        self.d2 = Dense(512, activation="relu")
        self.d3 = Dense(256, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(5)

    def call(self, x, meta, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        # z = tf.concat([x, meta], 1)  # Late combining of the metadata.
        # x = self.d4(z)
        x = self.d4(x)
        x = self.d5(x)

        return x

    def _convolution_step(self, kernel_size, channels):
        conv2d_1 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_2 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        conv2d_3 = Conv2D(channels, kernel_size=kernel_size, activation="relu")
        max_pool = MaxPooling2D(pool_size=(2, 2))

        return Sequential([conv2d_1, conv2d_2, conv2d_3, max_pool])

    def config(self, training=False, dry_run=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 1
        config.features = ALL_FEATURES
        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore
        if dry_run:
            config.error_strategy = dataloader.ErrorStrategy.stop
        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda target_ghi, metadata, image, target_cloudiness, timestamp, location: (
                target_ghi,
                metadata,
                self.scaling_image.normalize(image),
                target_cloudiness,
                timestamp,
                location
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model definition compatible with TensorFlow's eager execution.
Reference [Deep Residual Learning for Image
Recognition](https://arxiv.org/abs/1512.03385)
Adapted from tf.keras.applications.ResNet50. A notable difference is that the
model here outputs logits while the Keras model outputs probability.
"""


layers = tf.keras.layers


class _IdentityBlock(tf.keras.Model):
    """_IdentityBlock is the block that has no conv layer at shortcut.
    Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
    """

    def __init__(self, kernel_size, filters, stage, block, data_format):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
            filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
    """_ConvBlock is the block that has a conv layer at shortcut.
    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
          'channels_last').
        strides: strides for the convolution. Note that from stage 3, the first
         conv layer at main path is with strides=(2,2), and the shortcut should
         have strides=(2,2) as well.
    """

    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 data_format,
                 strides=(2, 2)):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
            filters1, (1, 1),
            strides=strides,
            name=conv_name_base + '2a',
            data_format=data_format)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b',
            data_format=data_format)
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

        self.conv_shortcut = layers.Conv2D(
            filters3, (1, 1),
            strides=strides,
            name=conv_name_base + '1',
            data_format=data_format)
        self.bn_shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


# Official Resnet model. Slighlty tweaked.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/benchmarks/resnet50/resnet50.py

# pylint: disable=not-callable
class ResNet50(base.Model):
    """Instantiates the ResNet50 architecture.
    Args:
      data_format: format for the image. Either 'channels_first' or
        'channels_last'.  'channels_first' is typically faster on GPUs while
        'channels_last' is typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
      name: Prefix applied to names of variables created in the model.
      trainable: Is the model trainable? If true, performs backward
          and optimization after call() method.
      include_top: whether to include the fully-connected layer at the top of the
        network.
      pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor
            output of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output of
            the last convolutional layer, and thus the output of the model will be
            a 2D tensor.
        - `max` means that global max pooling will be applied.
      block3_strides: whether to add a stride of 2 to block3 to make it compatible
        with tf.slim ResNet implementation.
      average_pooling: whether to do average pooling of block4 features before
        global pooling.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True.
    Raises:
        ValueError: in case of invalid argument for data_format.
    """

    def __init__(self,
                 data_format='channels_first',
                 name='',
                 trainable=True,
                 include_top=True,
                 pooling=None,
                 block3_strides=False,
                 average_pooling=True,
                 classes=5):
        super(ResNet50, self).__init__(title="Resnet50")

        self.scaling_image = preprocessing.MinMaxScaling(
            preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
        )
        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' %
                             (data_format, valid_channel_values))
        self.include_top = include_top
        self.block3_strides = block3_strides
        self.average_pooling = average_pooling
        self.pooling = pooling

        def conv_block(filters, stage, block, strides=(2, 2)):
            return _ConvBlock(
                3,
                filters,
                stage=stage,
                block=block,
                data_format=data_format,
                strides=strides)

        def id_block(filters, stage, block):
            return _IdentityBlock(
                3, filters, stage=stage, block=block, data_format=data_format)

        self.conv1 = layers.Conv2D(
            64, (7, 7),
            strides=(2, 2),
            data_format=data_format,
            padding='same',
            name='conv1')
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.max_pool = layers.MaxPooling2D((3, 3),
                                            strides=(2, 2),
                                            data_format=data_format)

        self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.l2b = id_block([64, 64, 256], stage=2, block='b')
        self.l2c = id_block([64, 64, 256], stage=2, block='c')

        self.l3a = conv_block([128, 128, 512], stage=3, block='a')
        self.l3b = id_block([128, 128, 512], stage=3, block='b')
        self.l3c = id_block([128, 128, 512], stage=3, block='c')
        self.l3d = id_block([128, 128, 512], stage=3, block='d')

        self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
        self.l4b = id_block([256, 256, 1024], stage=4, block='b')
        self.l4c = id_block([256, 256, 1024], stage=4, block='c')
        self.l4d = id_block([256, 256, 1024], stage=4, block='d')
        self.l4e = id_block([256, 256, 1024], stage=4, block='e')
        self.l4f = id_block([256, 256, 1024], stage=4, block='f')

        # Striding layer that can be used on top of block3 to produce feature maps
        # with the same resolution as the TF-Slim implementation.
        if self.block3_strides:
            self.subsampling_layer = layers.MaxPooling2D((1, 1),
                                                         strides=(2, 2),
                                                         data_format=data_format)
            self.l5a = conv_block([512, 512, 2048],
                                  stage=5,
                                  block='a',
                                  strides=(1, 1))
        else:
            self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
        self.l5b = id_block([512, 512, 2048], stage=5, block='b')
        self.l5c = id_block([512, 512, 2048], stage=5, block='c')

        self.avg_pool = layers.AveragePooling2D((7, 7),
                                                strides=(7, 7),
                                                data_format=data_format)

        if self.include_top:
            self.flatten = layers.Flatten()
            self.fc1000 = layers.Dense(classes, name='fc1000')
        else:
            reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
            reduction_indices = tf.constant(reduction_indices)
            if pooling == 'avg':
                self.global_pooling = functools.partial(
                    tf.reduce_mean,
                    reduction_indices=reduction_indices,
                    keep_dims=False)
            elif pooling == 'max':
                self.global_pooling = functools.partial(
                    tf.reduce_max, reduction_indices=reduction_indices, keep_dims=False)
            else:
                self.global_pooling = None

    def call(self, inputs, clearsky, training=True, intermediates_dict=None):
        """Call the ResNet50 model.
        Args:
          inputs: Images to compute features for.
          training: Whether model is in training phase.
          intermediates_dict: `None` or dictionary. If not None, accumulate feature
            maps from intermediate blocks into the dictionary.
            ""
        Returns:
          Tensor with featuremap.
        """

        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        if intermediates_dict is not None:
            intermediates_dict['block0'] = x

        x = self.max_pool(x)
        if intermediates_dict is not None:
            intermediates_dict['block0mp'] = x

        # Block 1 (equivalent to "conv2" in Resnet paper).
        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)
        if intermediates_dict is not None:
            intermediates_dict['block1'] = x

        # Block 2 (equivalent to "conv3" in Resnet paper).
        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)
        if intermediates_dict is not None:
            intermediates_dict['block2'] = x

        # Block 3 (equivalent to "conv4" in Resnet paper).
        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)
        x = self.l4f(x, training=training)

        if self.block3_strides:
            x = self.subsampling_layer(x)
            if intermediates_dict is not None:
                intermediates_dict['block3'] = x
        else:
            if intermediates_dict is not None:
                intermediates_dict['block3'] = x

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)
        x = self.l5c(x, training=training)

        if self.average_pooling:
            x = self.avg_pool(x)
            if intermediates_dict is not None:
                intermediates_dict['block4'] = x
        else:
            if intermediates_dict is not None:
                intermediates_dict['block4'] = x

        if self.include_top:
            return self.fc1000(self.flatten(x))
        elif self.global_pooling:
            return self.global_pooling(x)
        else:
            return x

    def config(self, training=False, dry_run=False) -> dataloader.DataloaderConfig:
        """Configuration."""
        config = default_config()
        config.num_images = 1
        config.ratio = 1
        config.features = ALL_FEATURES
        if training:
            config.error_strategy = dataloader.ErrorStrategy.skip
        else:
            config.error_strategy = dataloader.ErrorStrategy.ignore
        if dry_run:
            config.error_strategy = dataloader.ErrorStrategy.stop
        return config

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Applies the preprocessing to the inputs and the targets."""
        return dataset.map(
            lambda target_ghi, metadata, image, target_cloudiness, timestamp, location: (
                target_ghi,
                metadata,
                self.scaling_image.normalize(image),
                target_cloudiness,
                timestamp,
                location
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def _preprocess_target(self, target_ghi: tf.Tensor) -> tf.Tensor:
        return target_ghi[0:1]
