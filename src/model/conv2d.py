from datetime import datetime

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.optimizers import SGD

from src import logging
from src.data.train import load_data

logger = logging.create_logger(__name__)


def create_model():
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


def train(model, batch_size=32):
    logger.info("Training Conv2D model.")
    train_set, valid_set, _ = load_data()

    optimizer = SGD(0.0001)
    model.compile(
        loss="root_mean_squared_error",
        optimizer=optimizer,
        metrics=["root_mean_squared_error"],
    )

    log_directory = "/project/cq-training-1/project1/teams/team10/result_log" + datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = TensorBoard(log_dir=log_directory, histogram_freq=1)

    model.fit_generator(
        train_set.batch(batch_size),
        validation_data=valid_set.batch(batch_size),
        callbacks=[tensorboard_callback],
        epochs=1,
    )
    logger.info("Done.")
