import unittest

import numpy as np
import tensorflow as tf

from src.model import seq2seq


class LanguageModelTest(unittest.TestCase):
    def setUp(self):
        self.num_images = 3
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

        def encoder(input_images, training=False):
            return input_images

        self.encoder = encoder

    def test_gru_call(self):
        model = seq2seq.Gru(
            self.encoder,
            num_images=self.num_images,
            num_features=self.x * self.y * self.num_channels,
        )
        generated = model((self.images,))

        self.assertEqual(generated.shape, self.images.shape)

    def test_convlstm_call(self):
        model = seq2seq.ConvLSTM(
            self.encoder, num_images=self.num_images, num_channels=self.num_channels,
        )
        generated = model((self.images,))

        self.assertEqual(generated.shape, self.images.shape)

    def test_perdict_next_images(self):
        model = seq2seq.ConvLSTM(
            self.encoder, num_images=self.num_images, num_channels=self.num_channels,
        )

        num_generated = 3
        # Call predict_next_images without batch_size
        generated = model.predict_next_images(self.images[0], num_images=num_generated)

        self.assertEqual(
            generated.shape, (num_generated, self.x, self.y, self.num_channels)
        )

    def test_givenBlackImages_shouldPerdictNextImages(self):
        model = seq2seq.ConvLSTM(
            self.encoder, num_images=self.num_images, num_channels=self.num_channels,
        )

        num_generated = 3
        zeros = np.zeros((self.batch_size, 1, self.x, self.y, self.num_channels))
        images = tf.concat([self.images, zeros], 1)
        # Call predict_next_images without batch_size
        generated = model.predict_next_images(images[0], num_images=num_generated)

        self.assertEqual(
            generated.shape, (num_generated, self.x, self.y, self.num_channels)
        )
