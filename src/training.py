from datetime import datetime

import tensorflow as tf

from src import logging
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
CHECKPOINT_TIMESTAMP = 5


class SupervisedTraining:
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
        self.train_loss = tf.keras.metrics.Mean("train loss", dtype=tf.float32)
        self.valid_loss = tf.keras.metrics.Mean("valid loss", dtype=tf.float32)

    def run(self, batch_size=128, epochs=10, valid_batch_size=256, caching=False):
        """Performs the training of the model in minibatch.

        Agrs:
            batch_size:  parameter that determines the minibatch size
            epochs: a number of epochs
            valid_batch_size: should be as large as the GPU can handle.
            caching: if temporary caching is desired.
        """
        logger.info(f"Starting supervised training with model {self.model.title}")
        config = self.model.config(training=True)
        train_set, valid_set, _ = load_data(enable_tf_caching=caching, config=config)

        logger.info("Apply Preprocessing")
        train_set = self.model.preprocess(train_set)
        valid_set = self.model.preprocess(valid_set)

        logger.info("Creating loss logs")
        train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
        valid_summary_writer = tf.summary.create_file_writer(VALID_LOG_DIR)

        logger.info("Fitting model.")
        for epoch in range(epochs):
            logger.info("SupervisedTraining...")
            for i, (inputs, targets) in enumerate(train_set.batch(batch_size)):
                self._train_step(inputs, targets, training=True)
                logger.info(f"Batch #{i+1}")
            with train_summary_writer.as_default():
                tf.summary.scalar("train loss", self.train_loss.result(), step=epoch)

            logger.info("Evaluating validation loss")
            for inputs, targets in valid_set.batch(valid_batch_size):
                self._valid_step(inputs, targets, training=False)
            with valid_summary_writer.as_default():
                tf.summary.scalar("valid loss", self.valid_loss.result(), step=epoch)

            logger.info(
                f"Epoch: {epoch + 1}, Train loss: {self.train_loss.result()}, Valid loss: {self.valid_loss.result()} "
            )

            # Reset the cumulative metrics after each epoch
            self.train_loss.reset_states()
            self.valid_loss.reset_states()

            if epoch % CHECKPOINT_TIMESTAMP == 0:
                logger.info("Checkpointing...")
                self.model.save(str(epoch))

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
