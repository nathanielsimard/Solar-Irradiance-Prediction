import itertools
import os
from typing import Callable, Iterator, Tuple
from datetime import datetime

import tensorflow as tf

from src import logging
from src.data import dataloader, split, preprocessing
from src.data.metadata import Coordinates, Metadata, MetadataLoader, Station

logger = logging.create_logger(__name__)

STATION_COORDINATES = {
    Station.BND: Coordinates(40.05192, -88.37309, 230),
    Station.TBL: Coordinates(40.12498, -105.23680, 1689),
    Station.DRA: Coordinates(36.62373, -116.01947, 1007),
    Station.FPK: Coordinates(48.30783, -105.10170, 634),
    Station.GWN: Coordinates(34.25470, -89.87290, 98),
    Station.PSU: Coordinates(40.72012, -77.93085, 376),
    Station.SXF: Coordinates(43.73403, -96.62328, 473),
}

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
TRAIN_LOG_DIR = (
    "/project/cq-training-1/project1/teams/team10/tensorboard/run-"
    + CURRENT_TIME
    + "/train"
)
VALID_LOG_DIR = (
    "/project/cq-training-1/project1/teams/team10/tensorboard/run-"
    + CURRENT_TIME
    + "/valid"
)
MODEL_SAVE_DIR = "/project/cq-training-1/project1/teams/team10/models"
CHECKPOINT_TIMESTAMP = 5


class Training:
    """Training class."""

    def __init__(
        self,
        optimizer: tf.keras.optimizers,  # an optimizer object
        model: tf.keras.Model,  # a model object, should have a str method and a call method
        loss_fn: tf.keras.losses,  # a loss object
    ):
        """Initialize a training session."""
        self.optim = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.train_loss = tf.keras.metrics.Mean("train loss", dtype=tf.float32)
        self.valid_loss = tf.keras.metrics.Mean("valid loss", dtype=tf.float32)

        logger.info(model.summary())

    def training(self, batch_size=128, epochs=10, valid_batch_size=256, caching=False):
        """Performs the training of the model in minibatch.

        Params:
        A batch_size parameter that determines the minibatch size
        A number of epochs
        A valid_batch_size. Should be as large as the GPU can handle.
        A boolean caching variable if temporary caching is desired.
        """
        logger.info("Training" + str(self.model) + "model.")
        train_set, valid_set, _ = load_data(enable_tf_caching=caching)

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

        logger.info("Creating loss logs")
        train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
        valid_summary_writer = tf.summary.create_file_writer(VALID_LOG_DIR)

        logger.info("Fitting model.")
        for epoch in range(epochs):
            i = 0
            logger.info("Training...")
            for inputs, targets in train_set.batch(batch_size):
                self._train_step(inputs, targets, training=True)
                i += 1
                print("Batch #", i)
            with train_summary_writer.as_default():
                tf.summary.scalar("train loss", self.train_loss.result(), step=epoch)

            logger.info("Evaluating validation loss")
            for inputs, targets in valid_set.batch(valid_batch_size):
                self._valid_step(inputs, targets, training=False)
            with valid_summary_writer.as_default():
                tf.summary.scalar("valid loss", self.valid_loss.result(), step=epoch)

            # From the Tensorflow documentation
            template = (
                "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
            )
            print(
                template.format(
                    epoch + 1, self.train_loss.result(), self.valid_loss.result(),
                )
            )
            # Reset the cumulative metrics after each epoch
            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            if epoch % CHECKPOINT_TIMESTAMP == 0:
                logger.info("Checkpointing...")
                self.model.save(
                    filepath=MODEL_SAVE_DIR + str(self.model),
                    save_format="tf",
                    overwrite=True,
                )

        logger.info("Done.")

    @tf.function
    def _train_step(self, train_inputs, train_targets, training: bool):
        with tf.GradientTape() as tape:
            outputs = self.model(train_inputs, training)
            loss = self.loss_fn(train_targets, outputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    @tf.function
    def _valid_step(self, valid_inputs, valid_targets, training: bool):
        outputs = self.model(valid_inputs, training)
        loss = self.loss_fn(valid_targets, outputs)

        self.valid_loss(loss)


def default_cache_dir():
    """Use SCRATCH directory on helios, tmp otherwise."""
    try:
        return os.environ["SCRATCH"]
    except KeyError:
        return "/tmp"


def default_config():
    """Default training configurations."""
    return dataloader.Config(
        error_strategy=dataloader.ErrorStrategy.skip,
        crop_size=(64, 64),
        image_cache_dir=default_cache_dir(),
        features=[dataloader.Feature.image, dataloader.Feature.target_ghi],
        channels=["ch1", "ch2", "ch3", "ch4", "ch6"],
    )


def load_data(
    file_name="/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
    night_time=False,
    skip_missing=True,
    config=default_config(),
    enable_tf_caching=False,
    cache_file=default_cache_dir(),
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, valid and test datasets.

    Return: (train_dataset, valid_dataset, test_dataset)
    """
    train_datetimes, valid_datetimes, test_datetimes = split.load()

    metadata_loader = MetadataLoader(file_name=file_name)
    metadata_train = metadata_station(
        metadata_loader,
        train_datetimes,
        night_time=night_time,
        skip_missing=skip_missing,
    )
    metadata_valid = metadata_station(
        metadata_loader,
        valid_datetimes,
        night_time=night_time,
        skip_missing=skip_missing,
    )
    metadata_test = metadata_station(
        metadata_loader,
        test_datetimes,
        night_time=night_time,
        skip_missing=skip_missing,
    )

    dataset_train = dataloader.create_dataset(metadata_train, config)
    dataset_valid = dataloader.create_dataset(metadata_valid, config)
    dataset_test = dataloader.create_dataset(metadata_test, config)

    if enable_tf_caching:
        dataset_train = dataset_train.cache(cache_file + "_train")
        dataset_test = dataset_test.cache(cache_file + "_test")
        dataset_valid = dataset_valid.cache(cache_file + "_valid")

    logger.info("Loaded datasets.")
    return dataset_train, dataset_valid, dataset_test


def metadata_station(
    metadata_loader, datetimes, night_time=False, skip_missing=True
) -> Callable[[], Iterator[Metadata]]:
    """Create metadata for all stations."""

    def gen():
        generators = []
        for station, coordinate in STATION_COORDINATES.items():
            generators.append(
                metadata_loader.load(
                    station,
                    coordinate,
                    night_time=night_time,
                    target_datetimes=datetimes,
                    skip_missing=skip_missing,
                )
            )
        return itertools.chain(*generators)

    return gen


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
