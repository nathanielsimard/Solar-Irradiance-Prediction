import logging

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.optimizers import SGD

from src.data.train import load_data

logger = logging.getLogger(__name__)


def create_model():
    input_shape = (64, 64, 5)
    model = Sequential(
        [
            Conv2D(32, kernel_size=(8, 8), input_shape=input_shape),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(8, 8)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128),
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
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mse"])

    historic = model.fit_generator(
        train_set.batch(batch_size),
        validation_data=valid_set.batch(batch_size),
        epochs=1,
    )
    logger.info("Done.")
    print(historic)
