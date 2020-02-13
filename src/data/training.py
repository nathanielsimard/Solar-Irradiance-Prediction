import tensorflow as tf

from src import logging
from src.data.train import load_data, load_data_and_create_generators
from src.data import preprocessing
from src import env
from datetime import datetime
import time as time

logger = logging.create_logger(__name__)


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
        self.CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        self.TRAIN_LOG_DIR = env.get_tensorboard_log_directory() + "/train"
        self.VALID_LOG_DIR = env.get_tensorboard_log_directory() + "/valid"
        self.MODEL_SAVE_DIR = env.get_model_checkpoint_directory()
        self.CHECKPOINT_TIMESTAMP = 5

    def checkpoint(self):
        if self.enable_checkpoint:
            self.model.save(
                filepath=self.MODEL_SAVE_DIR + str(self.model),
                save_format="tf",
                overwrite=True,
            )

    def run(
        self,
        batch_size=128,
        epochs=10,
        valid_batch_size=256,
        enable_tf_caching=False,
        dry_run=False,
        skip_non_cached=False,
        enable_checkpoint=True
    ):
        """Performs the training of the model in minibatch.

        Agrs:
            batch_size:  parameter that determines the minibatch size
            epochs: a number of epochs
            valid_batch_size: should be as large as the GPU can handle.
            caching: if temporary caching is desired.
        """
        self.enable_checkpoint = enable_checkpoint
        logger.info("Training" + str(self.model) + "model.")
        train_set, valid_set, _ = load_data(
            enable_tf_caching=enable_tf_caching, skip_non_cached=skip_non_cached
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

        if dry_run:
            # Only test the generators, for debugging weird behavior and corner cases.
            (
                train_generator,
                valid_generator,
                test_generator,
            ) = load_data_and_create_generators(
                enable_tf_caching=enable_tf_caching, skip_non_cached=skip_non_cached
            )
            for sample in train_generator:
                print(
                    sample
                )  # Just make sure that we can get a single sample out of the dry-run
                break

        logger.info("Creating loss logs")
        train_summary_writer = tf.summary.create_file_writer(self.TRAIN_LOG_DIR)
        valid_summary_writer = tf.summary.create_file_writer(self.VALID_LOG_DIR)

        logger.info("Fitting model.")
        begin = time.time()
        self.checkpoint()  # Fail early!
        for epoch in range(epochs):
            logger.info("Training...")
            for i, (inputs, targets) in enumerate(train_set.batch(batch_size)):
                self._train_step(inputs, targets, training=True)
                sps = batch_size / (time.time() - begin)
                logger.info(
                    f"Batch #{i+1}, size={batch_size}, samples per seconds={sps}"
                )
                begin = time.time()
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
            if epoch % self.CHECKPOINT_TIMESTAMP == 0:
                logger.info("Checkpointing...")
                self.checkpoint()
                # self.model.save(
                #    filepath=self.MODEL_SAVE_DIR + str(self.model),
                #    save_format="tf",
                #    overwrite=True,
                # )

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
