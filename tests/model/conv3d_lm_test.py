import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from src.model import conv3d_lm, languagemodel


class Conv3DTest(unittest.TestCase):
    def setUp(self):
        self.num_images = 6
        self.num_channels = 5
        self.x = 3
        self.y = 3
        self.batch_size = 2

        self.images = tf.constant(
            np.random.random(
                (self.batch_size, self.num_images, self.x, self.y, self.num_channels)
            ),
            dtype=tf.float32,
        )

        language_model = mock.MagicMock(languagemodel.LanguageModel)

        def predict_next_images(images, num_images=6):
            return images[:num_images]

        language_model.predict_next_images = predict_next_images
        self.model = conv3d_lm.Conv3D(language_model)

    def test_call(self):
        def gen():
            for images in self.images:
                yield (
                    images,
                    tf.constant([100, 100, 100, 100]),
                    tf.constant([100, 100, 100, 100]),
                )

        dataset = tf.data.Dataset.from_generator(
            gen, (tf.float32, tf.float32, tf.float32)
        )
        dt = self.model.preprocess(dataset)
        for d in dt:
            continue
