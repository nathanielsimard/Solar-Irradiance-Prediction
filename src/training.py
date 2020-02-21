import pickle
from datetime import datetime

import tensorflow as tf

from src import env, logging
from src.data.train import load_data
from src.model.base import Model

logger = logging.create_logger(__name__)

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
TEST_LOG_DIR = (
    "/project/cq-training-1/project1/teams/team10/tensorboard/run-"
    + CURRENT_TIME
    + "/test"
)
CHECKPOINT_TIMESTAMP = 5


class History(object):
    """Keeps track of the different losses."""

    def __init__(self):
        """Initialize dictionaries."""
        self.logs = {"train": [], "valid": [], "test": []}

    def record(self, name, value):
        """Stores value in the corresponding log."""
        self.logs[name].append(value)

    def save(self, file_name):
        """Save file."""
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_name):
        """Load file."""
        with open(file_name, "rb") as file:
            return pickle.load(file)


class Training(object):
    """Train a model in a supervised way.

    It assumes that the data is labeled (inputs, targets).
    """

    def __init__(
        self, optimizer: tf.keras.optimizers, model: Model, loss_fn: tf.keras.losses,
    ):
        """Initialize a training session."""
        self.optim = optimizer
        self.loss_fn = loss_fn
        self.model = model

        self.metrics = {
            "train": tf.keras.metrics.Mean("train loss", dtype=tf.float32),
            "valid": tf.keras.metrics.Mean("valid loss", dtype=tf.float32),
            "test": tf.keras.metrics.Mean("test loss", dtype=tf.float32),
        }
        self.writer = {
            "train": tf.summary.create_file_writer(
                env.get_tensorboard_log_directory() + "/train/"
            ),
            "valid": tf.summary.create_file_writer(
                env.get_tensorboard_log_directory() + "/valid/"
            ),
            "test": tf.summary.create_file_writer(
                env.get_tensorboard_log_directory() + "/test/"
            ),
        }

        self.history = History()

    def run(
        self,
        batch_size=128,
        epochs=25,
        valid_batch_size=128,
        enable_tf_caching=False,
        skip_non_cached=False,
        enable_checkpoint=True,
        dry_run=False,
        categorical=False,
    ):
        """Performs the training of the model in minibatch.

        Agrs:
            batch_size:  parameter that determines the minibatch size
            epochs: a number of epochs
            valid_batch_size: should be as large as the GPU can handle.
            caching: if temporary caching is desired.
        """
        config = self.model.config(training=True)
        logger.info(
            f"Starting training\n"
            + f" - Model: {self.model.title}\n"
            + f" - Config: {config}"
        )

        train_set, valid_set, test_set = load_data(
            enable_tf_caching=enable_tf_caching,
            config=config,
            skip_non_cached=skip_non_cached,
        )

        logger.info("Apply Preprocessing")
        train_set = self.model.preprocess(train_set)
        valid_set = self.model.preprocess(valid_set)
        test_set = self.model.preprocess(test_set)

        logger.info("Creating loss logs")

        # Fail early!
        self.model.save(str(0))
        self._evaluate("test", 0, test_set, valid_batch_size, dry_run=True)
        logger.info("Fitting model.")
        for epoch in range(epochs):
            logger.info("Supervised training...")

            for i, data in enumerate(train_set.batch(batch_size)):
                inputs = data[:-1]
                targets = data[-1]
                logger.info(f"Batch #{i+1}")

                self._train_step(inputs, targets)

            logger.info("Evaluating validation loss")
            self._evaluate("valid", epoch, valid_set, valid_batch_size)

            logger.info("Checkpointing...")
            self.model.save(str(epoch))

            self._update_progress(epoch)
            self.history.save(f"{self.model.title}-{epoch}")

        self._evaluate("test", epoch, test_set, valid_batch_size)
        logger.info("Done.")

    def _update_progress(self, epoch):
        train_metric = self.metrics["train"]
        valid_metric = self.metrics["valid"]
        train_writer = self.writer["train"]

        logger.info(
            f"Epoch: {epoch + 1}, Train loss: {train_metric.result()}, Valid loss: {valid_metric.result()} "
        )

        with train_writer.as_default():
            tf.summary.scalar("train", train_metric.result(), step=epoch)

        # Reset the cumulative metrics after each epoch
        self.history.record("train", train_metric.result())
        train_metric.reset_states()
        valid_metric.reset_states()

    def _evaluate(self, name, epoch, dataset, batch_size, dry_run=False):
        metric = self.metrics[name]
        writer = self.writer[name]

        for i, data in enumerate(dataset.batch(batch_size)):
            logger.info(f"Evaluation batch #{i}")
            inputs = data[:-1]
            targets = data[-1]

            loss = self._calculate_loss(inputs, targets)
            metric(loss)
            if dry_run:
                break

        with writer.as_default():
            tf.summary.scalar(name, metric.result(), step=epoch)

        self.history.record(name, metric.result())

    @tf.function
    def _train_step(self, train_inputs, train_targets):
        with tf.GradientTape() as tape:
            outputs = self.model(train_inputs, training=True)
            loss = self.loss_fn(train_targets, outputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.metrics["train"](loss)

    @tf.function
    def _calculate_loss(self, valid_inputs, valid_targets):
        outputs = self.model(valid_inputs)
        return self.loss_fn(valid_targets, outputs)
