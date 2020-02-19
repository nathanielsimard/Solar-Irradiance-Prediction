import pickle
from datetime import datetime

import tensorflow as tf
import numpy as np

from src import logging
from src.data.train import load_data
from src.model.base import Model
from src import env

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


class SupervisedTraining(object):
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
        self.train_rmse = tf.keras.metrics.RootMeanSquaredError()
        self.valid_rmse = tf.keras.metrics.RootMeanSquaredError()
        # self.train_accuracy = tf.keras.metrics.Accuracy()
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.valid_accuracy = tf.keras.metrics.CategoricalAccuracy()

        self.metrics = {
            "train": tf.keras.metrics.Mean("train loss", dtype=tf.float32),
            "train_rmse": tf.keras.metrics.RootMeanSquaredError(),
            "valid": tf.keras.metrics.Mean("valid loss", dtype=tf.float32),
            "valid_rmse": tf.keras.metrics.RootMeanSquaredError(),
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
        epochs=10,
        valid_batch_size=256,
        enable_tf_caching=False,
        skip_non_cached=False,
        enable_checkpoint=True,
        dry_run=False,
        categorical=True

    ):
        """Performs the training of the model in minibatch.

        Agrs:
            batch_size:  parameter that determines the minibatch size
            epochs: a number of epochs
            valid_batch_size: should be as large as the GPU can handle.
            caching: if temporary caching is desired.
        """
        self.categorical = categorical
        logger.info(f"Starting supervised training with model {self.model.title}")
        config = self.model.config(training=True, dry_run=dry_run)
        train_set, valid_set, test_set = load_data(
            enable_tf_caching=enable_tf_caching,
            dataloader_config=config,
            skip_non_cached=skip_non_cached,
        )

        logger.info("Apply Preprocessing")
        #train_set = self.model.preprocess(train_set)
        #valid_set = self.model.preprocess(valid_set)
        #test_set = self.model.preprocess(valid_set)

        logger.info("Creating loss logs")

        # Fail early!
        self.model.save(str(0))
        self._evaluate("test", 0, test_set, valid_batch_size, dry_run=True)

        # epoch_output = np.array()
        epoch_results = None

        logger.info("Fitting model.")
        for epoch in range(epochs):
            logger.info("Supervised training...")
            if dry_run:
                self._evaluate("valid", epoch, valid_set, valid_batch_size)

            for i, (targets, meta, image, target_cloudiness, timestamp, location) in enumerate(train_set.batch(batch_size)):
                logger.info(f"Batch #{i+1}")
                clearsky = meta[:, 0:4]  # TODO: Do something better.
                # clearsky = clearsky / 1000  # Cheap normalization
                adjusted_target = tf.clip_by_value(targets, 0, 2000) / tf.clip_by_value(clearsky, 1, 2000)

                train_step_results = self._train_step(adjusted_target, targets, clearsky, image,
                                                      target_cloudiness, timestamp, location, training=True)
                if epoch_results is None:
                    epoch_results = train_step_results
                else:
                    # Highly inefficient, but should not slow down training.
                    epoch_results = np.append(epoch_results, train_step_results, axis=0)
                if (i % 10 == 0):
                    self._update_progress(i)

            np.save(f"epoch_results{epoch}.npy", epoch_results)

            epoch_results = None  # Reset at the end of one epoch
            logger.info("Evaluating validation loss")
            self._evaluate("valid", epoch, valid_set, valid_batch_size)

            if enable_checkpoint and epoch % CHECKPOINT_TIMESTAMP == 0:
                logger.info("Checkpointing...")
                self.model.save(str(epoch))

            self._update_progress(epoch)

        self._evaluate("test", epoch, test_set, valid_batch_size)

        self.history.save(f"{self.model.title}-{epochs}")
        logger.info("Done.")

    def _update_progress(self, epoch):
        train_metric = self.metrics["train"]
        valid_metric = self.metrics["valid"]
        train_writer = self.writer["train"]

        logger.info(
            f"Step: {epoch + 1}, Train acc: {self.train_accuracy.result()} Train loss: {train_metric.result()}, Train RMSE: {self.train_rmse.result()}, Valid loss: {valid_metric.result()} "
        )

        with train_writer.as_default():
            tf.summary.scalar("train", train_metric.result(), step=epoch)

        # Reset the cumulative metrics after each epoch
        self.history.record("train", train_metric.result())
        train_metric.reset_states()
        valid_metric.reset_states()
        self.train_rmse.reset_states()
        self.valid_rmse.reset_states()
        self.train_accuracy.reset_states()

    def _evaluate(self, name, epoch, dataset, batch_size, dry_run=False):
        metric = self.metrics[name]
        writer = self.writer[name]
        for targets, meta, image, target_cloudiness, timestamp, location in dataset.batch(batch_size):
            clearsky = meta[:, 0:4]  # TODO: Do something better.
            # clearsky = clearsky / 1000  # Cheap normalization
            adjusted_target = tf.clip_by_value(targets, 0, 2000) / tf.clip_by_value(clearsky, 1, 2000)

            loss = self._valid_step(adjusted_target, targets, clearsky, image,
                                    target_cloudiness, timestamp, location, training=True)
            metric(loss)
            if dry_run:
                break

        with writer.as_default():
            tf.summary.scalar(name, metric.result(), step=epoch)

        self.history.record(name, metric.result())

    # @tf.function
    def _categorical_to_ghi(self, outputs, clearsky):
        outputs_onehot = tf.one_hot(tf.argmax(outputs, 1), depth=5)
        penalty = outputs_onehot[:, 3] * 0.99 + outputs_onehot[:, 4] * \
            0.75 + outputs_onehot[:, 2] * 0.85 + outputs_onehot[:, 1] * 0.45 + outputs_onehot[:, 0] * 0
        # no_penalty = outputs_onehot[:, 3] * 1 + outputs_onehot[:, 4] * \
        #    1 + outputs_onehot[:, 2] * 1 + outputs_onehot[:, 1] * 1 + outputs_onehot[:, 0] * 1
        clearsky_t0 = clearsky[:, 0] * penalty
        clearsky_t1 = clearsky[:, 1] * penalty
        clearsky_t3 = clearsky[:, 2] * penalty
        clearsky_t6 = clearsky[:, 3] * penalty

        output_ghi = tf.stack([clearsky_t0, clearsky_t1, clearsky_t3, clearsky_t6], axis=1)
        return (output_ghi, outputs_onehot, penalty)

    def _train_step(self, adjusted_targets, targets, clearsky, image, target_cloudiness,
                    timestamp, location, training: bool):

        # if cloudiness == 'night':
        # return tmp_clearsky
        #    if cloudiness == 'cloudy':
        # return tmp_clearsky - (tmp_clearsky*0.5)
        # if cloudiness == 'slightly cloudy':
        #    return tmp_clearsky - (tmp_clearsky*0.25)
        # if cloudiness == 'clear':
        #    return tmp_clearsky
        # if cloudiness == 'variable':
        #    return tmp_clearsky - (tmp_clearsky*0.05)

        # pd.Series(timestamp.numpy()).apply(pd.Timestamp, unit='s')
        results = np.zeros(1)

        with tf.GradientTape() as tape:
            outputs = self.model(image, clearsky, training)
            # real_outputs = clearsky * outputs
            # self.last_outputs = outputs.numpy()
            # self.last_real_outputs = real_outputs.numpy()
            # self.train_rmse.update_state(targets, real_outputs)
            #        one_hot = {
            # "night": np.array([1, 0, 0, 0, 0]),
            # "cloudy": np.array([0, 1, 0, 0, 0]), 45%
            # "slightly cloudy": np.array([0, 0, 1, 0, 0]), 85%
            # "clear": np.array([0, 0, 0, 1, 0]), 99%
            # "variable" : np.array([0, 0, 0, 0, 1]), 75%
            # }
            if self.categorical:
                output_ghi, outputs_onehot, penalty = self._categorical_to_ghi(outputs, clearsky)
                self.train_accuracy.update_state(target_cloudiness, outputs)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_cloudiness, logits=outputs)
                results = tf.concat([tf.expand_dims(timestamp, axis=1), location, clearsky, outputs,
                                     outputs_onehot, target_cloudiness, tf.expand_dims(penalty, axis=1),
                                     targets, output_ghi], axis=1).numpy()

            else:
                output_ghi = outputs  # * clearsky

                #loss = tf.math.sqrt(tf.reduce_sum((targets - output_ghi)**2) / len(targets))
                loss = self.loss_fn(targets, outputs)

                results = tf.concat([tf.expand_dims(timestamp, axis=1), location, clearsky, outputs,
                                     target_cloudiness, adjusted_targets, targets, output_ghi], axis=1).numpy()

            self.train_rmse.update_state(targets, output_ghi)

            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #    labels=tf.argmax(target_cloudiness, 1), logits=outputs)

            # loss = self.loss_fn(adjusted_targets, outputs)  # By convention, the target will always come first
            np.save("current_results.npy", results)
            np.save("current_images.npy", image)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.metrics["train"](loss)
        return results

    # @tf.function
    def _valid_step(self, adjusted_targets, targets, clearsky, image, target_cloudiness,
                    timestamp, location, training: bool):
        outputs = self.model(image, clearsky, training)

        if self.categorical:
            output_ghi, outputs_onehot, penalty = self._categorical_to_ghi(outputs, clearsky)
            self.train_accuracy.update_state(target_cloudiness, outputs)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_cloudiness, logits=outputs)

        else:
            output_ghi = outputs * clearsky
            loss = self.loss_fn(adjusted_targets, outputs)

        self.valid_rmse.update_state(targets, output_ghi)

        return loss
