from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D

from src import logging

logger = logging.create_logger(__name__)


def create_model():
    """Create Conv2D model."""
    input_shape = (64, 64, 5)
    model = Sequential(
        [
            Conv2D(64, kernel_size=(5, 5), input_shape=input_shape),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(5, 5)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256),
            Activation("relu"),
            Dense(4),
        ]
    )
    logger.info(model.summary())
    return model
