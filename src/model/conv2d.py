from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.optimizers import SGD

from src import logging
from src.data import preprocessing
from src.data.train import load_data

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
    print(model.summary())
    return model


def train(model, batch_size=32, epochs=10):
    """Train Conv2D model."""
    logger.info("Training Conv2D model.")
    train_set, valid_set, _ = load_data(enable_tf_caching=True)

    scaling_image = preprocessing.MinMaxScaling(0, 255)
    scaling_target = preprocessing.MinMaxScaling(-6, 1200)

    train_set = _scale_dataset(scaling_image, scaling_target, train_set)
    valid_set = _scale_dataset(scaling_image, scaling_target, valid_set)

    optimizer = SGD(0.0001)
    model.compile(
        loss="mse", optimizer=optimizer, metrics=["mse"],
    )

    log_directory = "/project/cq-training-1/project1/teams/team10/result_log" + datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = TensorBoard(log_dir=log_directory, histogram_freq=1)

    model.fit_generator(
        train_set.batch(batch_size),
        validation_data=valid_set.batch(batch_size),
        callbacks=[tensorboard_callback],
        epochs=epochs,
    )
    logger.info("Done.")


def _scale_dataset(
    scaling_image: preprocessing.MinMaxScaling,
    scaling_target: preprocessing.MinMaxScaling,
    dataset: tf.data.Dataset,
):
    return dataset.map(
        lambda image, target_ghi: (
            scaling_image.normalize(image),
            scaling_target.normalize(target_ghi),
        )
    )
