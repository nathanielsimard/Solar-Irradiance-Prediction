import unittest
from unittest import mock

import tensorflow as tf

from src.data import preprocessing
from src.session import Session


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.scaling_ghi = preprocessing.min_max_scaling_ghi()
        self.session = Session(mock.Mock())

    def test_shi_rmse(self):
        predicted = tf.constant([950.0])
        target = tf.constant([850.0])
        expected = self.session.loss_fn(predicted, target).numpy()

        rescaled_predicted = self.session.scaling_ghi.normalize(predicted)
        rescaled_target = self.session.scaling_ghi.normalize(target)

        loss = self.session.loss_fn(rescaled_predicted, rescaled_target)
        actual = self.session._rescale_loss_ghi(loss).numpy()
        self.assertAlmostEqual(expected, actual, delta=1e-5)
