import unittest

import numpy as np

from src.model import languagemodel


class LanguageModelTest(unittest.TestCase):
    def setUp(self):
        self.num_images = 3
        self.num_channels = 5
        self.x = 3
        self.y = 3
        self.batch_size = 2
        self.images = np.random.random(
            (self.batch_size, self.num_images, self.x, self.y, self.num_channels)
        )

        def encoder(input_images, training=False):
            return input_images

        self.model = languagemodel.LanguageModel(
            encoder,
            num_images=self.num_images,
            num_features=self.x * self.y * self.num_channels,
        )

    def test_call(self):
        generated = self.model((self.images,))

        self.assertEqual(generated.shape, self.images.shape)

    def test_perdict_next_images(self):
        num_generated = 3
        # Call predict_next_images without batch_size
        generated = self.model.predict_next_images(
            self.images[0], num_images=num_generated
        )

        self.assertEqual(
            generated.shape, (num_generated, self.x, self.y, self.num_channels)
        )
