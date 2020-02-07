import tensorflow as tf

from src.data import dataloader
from src.data.train import default_config, load_data


class MinMaxScaling(object):
    """Scale value given the maximum and minimum possible values."""

    def __init__(self, min_value: float, max_value: float):
        """Create Min Max Scaler."""
        self.min_value = min_value
        self.max_value = max_value

    def normalize(self, value: tf.Tensor) -> tf.Tensor:
        """Normalize value.

        If the value is between the min and max, it will be between
        zero and one.
        """
        return (value - self.min_value) / (self.max_value - self.min_value)

    def original(self, value: tf.Tensor) -> tf.Tensor:
        """Return the original values of a scaled values."""
        return (value * (self.max_value - self.min_value)) + self.min_value


def find_target_ghi_min_value(dataset=None):
    """Find the minimum value of target ghi.

    The values are found based on the training dataset.
    """
    if dataset is None:
        config = default_config()
        config.features = [dataloader.Feature.target_ghi]
        dataset, _, _ = load_data(config=config)

    max_value = 100000.0  # Big enough, but still float32
    return dataset.reduce(max_value, _reduce_min)


def find_target_ghi_max_value(dataset=None):
    """Find the maximum value of target ghi.

    The values are found based on the training dataset.
    """
    if dataset is None:
        config = default_config()
        config.features = [dataloader.Feature.target_ghi]
        dataset, _, _ = load_data(config=config)

    return dataset.reduce(0.0, _reduce_max)


def _reduce_max(acc, x):
    max_x = tf.math.reduce_max(x[0], keepdims=False)
    return tf.math.maximum(acc, max_x)


def _reduce_min(acc, x):
    min_x = tf.math.reduce_min(x[0], keepdims=False)
    return tf.math.minimum(acc, min_x)
