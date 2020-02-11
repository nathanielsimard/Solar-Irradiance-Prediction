import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from src import logging
from src import env
from src.data import preprocessing
from src.data.train import load_data, load_data_and_create_generators

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


def train(model, batch_size=32, epochs=10, enable_tf_caching=False, dry_run=False):
    """Train Conv2D model."""
    logger.info("Training Conv2D model.")
    train_set, valid_set, _ = load_data(
        enable_tf_caching=enable_tf_caching, file_name=env.get_catalog_path()
    )

    scaling_image = preprocessing.MinMaxScaling(
        preprocessing.IMAGE_MIN, preprocessing.IMAGE_MAX
    )
    scaling_target = preprocessing.MinMaxScaling(
        preprocessing.TARGET_GHI_MIN, preprocessing.TARGET_GHI_MAX
    )

    logger.info("Scaling train set.")
    train_set = _scale_dataset(scaling_image, scaling_target, train_set)
    logger.info("Scaling valid set.")
    valid_set = _scale_dataset(scaling_image, scaling_target, valid_set)

    optimizer = SGD(0.0001)
    logger.info("Compiling model.")
    model.compile(
        loss="mse", optimizer=optimizer, metrics=["mse"],
    )

    tensorboard_callback = TensorBoard(
        log_dir=env.get_tensorboard_log_directory(),
        update_freq="batch",
        profile_batch=0,
    )

    logger.info("Fit model.")
    if not dry_run:
        model.fit_generator(
            train_set.batch(batch_size),
            validation_data=valid_set.batch(batch_size),
            callbacks=[tensorboard_callback],
            epochs=epochs,
        )
    else:
        train_set, valid_set, train_set = load_data_and_create_generators(
            enable_tf_caching=enable_tf_caching, file_name=env.get_catalog_path()
        )
        for sample in train_set:
            print(sample)

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
