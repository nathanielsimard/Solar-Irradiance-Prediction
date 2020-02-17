import pickle
from datetime import datetime

import tensorflow as tf

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
        #self.train_accuracy = tf.keras.metrics.Accuracy()
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
        dry_run=False
    ):
        """Performs the training of the model in minibatch.

        Agrs:
            batch_size:  parameter that determines the minibatch size
            epochs: a number of epochs
            valid_batch_size: should be as large as the GPU can handle.
            caching: if temporary caching is desired.
        """
        logger.info(f"Starting supervised training with model {self.model.title}")
        config = self.model.config(training=True, dry_run=dry_run)
        train_set, valid_set, test_set = load_data(
            enable_tf_caching=enable_tf_caching,
            dataloader_config=config,
            skip_non_cached=skip_non_cached,
        )

        logger.info("Apply Preprocessing")
        train_set = self.model.preprocess(train_set)
        valid_set = self.model.preprocess(valid_set)
        test_set = self.model.preprocess(valid_set)

        logger.info("Creating loss logs")

        # Fail early!
        self.model.save(str(0))

        logger.info("Fitting model.")
        for epoch in range(epochs):
            logger.info("Supervised training...")

            for i, (targets, meta, image, target_cloudiness) in enumerate(train_set.batch(batch_size)):
                logger.info(f"Batch #{i+1}")
                clearsky = meta[:, 0:4]  # TODO: Do something better.
                # clearsky = clearsky / 1000  # Cheap normalization
                adjusted_target = tf.clip_by_value(tf.math.abs(targets / clearsky), 0, 1)
                #adjusted_target = abs(clearsky - targets) / clearsky
                #adjusted_target[adjusted_target == float("Inf")] = 0

                self._train_step(adjusted_target, targets, clearsky, image, target_cloudiness, training=True)
                if (i % 10 == 0):
                    self._update_progress(i)

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

    def _evaluate(self, name, epoch, dataset, batch_size):
        metric = self.metrics[name]
        writer = self.writer[name]
        for targets, meta, image, target_cloudiness in dataset.batch(batch_size):
            clearsky = meta[:, 0:4]  # TODO: Do something better.
            loss = self._calculate_loss(image, clearsky, targets, training=False)
            metric(loss)

        with writer.as_default():
            tf.summary.scalar(name, metric.result(), step=epoch)

        self.history.record(name, metric.result())

    # @tf.function
    def _train_step(self, adjusted_targets, targets, clearsky, image, target_cloudiness, training: bool):

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

        with tf.GradientTape() as tape:
            outputs = self.model(image, clearsky, training)
            #real_outputs = clearsky * outputs
            #self.last_outputs = outputs.numpy()
            #self.last_real_outputs = real_outputs.numpy()
            #self.train_rmse.update_state(targets, real_outputs)
            #        one_hot = {
            # "night": np.array([1, 0, 0, 0, 0]),
            # "cloudy": np.array([0, 1, 0, 0, 0]),
            # "slightly cloudy": np.array([0, 0, 1, 0, 0]),
            # "clear": np.array([0, 0, 0, 1, 0]),
            # "variable" : np.array([0, 0, 0, 0, 1]),
            # }

            outputs_onehot = tf.one_hot(tf.argmax(outputs, 1), depth=5)
            #penalty = outputs_onehot[:, 3] * 1 + outputs_onehot[:, 4] * \
            #    0.95 + outputs_onehot[:, 2] * 0.75 + outputs_onehot[:, 1] * 0.5
            #clearsky_t0 = clearsky[:, 0] * penalty
            #clearsky_t1 = clearsky[:, 1] * penalty
            #clearsky_t3 = clearsky[:, 2] * penalty
            #clearsky_t6 = clearsky[:, 3] * penalty

            penalty = target_cloudiness[:, 3] * 1 + target_cloudiness[:, 4] * \
                0.95 + target_cloudiness[:, 2] * 0.75 + target_cloudiness[:, 1] * 0.5
            clearsky_t0 = clearsky[:, 0] * penalty
            clearsky_t1 = clearsky[:, 1] * penalty
            clearsky_t3 = clearsky[:, 2] * penalty
            clearsky_t6 = clearsky[:, 3] * penalty

            output_ghi = tf.stack([clearsky_t0, clearsky_t1, clearsky_t3, clearsky_t6], axis=1)
        

            self.train_accuracy.update_state(target_cloudiness, outputs)
            self.train_rmse.update_state(targets, output_ghi)

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_cloudiness, logits=outputs)
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #    labels=tf.argmax(target_cloudiness, 1), logits=outputs)

            # loss = self.loss_fn(adjusted_targets, outputs)  # By convention, the target will always come first
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.metrics["train"](loss)

    @tf.function
    def _calculate_loss(self, image, clearsky, valid_targets, target_cloudiness, training: bool):
        outputs = self.model(image, meta, training)
        outputs_onehot = tf.one_hot(tf.argmax(outputs, 1), depth=5)
        penalty = outputs_onehot[:, 3] * 1 + outputs_onehot[:, 4] * \
            0.95 + outputs_onehot[:, 2] * 0.75 + outputs_onehot[:, 1] * 0.5
        clearsky_t0 = clearsky[:, 0] * penalty
        clearsky_t1 = clearsky[:, 1] * penalty
        clearsky_t3 = clearsky[:, 2] * penalty
        clearsky_t6 = clearsky[:, 3] * penalty

        output_ghi = tf.stack([clearsky_t0, clearsky_t1, clearsky_t3, clearsky_t6], axis=1)

        self.valid_rmse.update_state(valid_targets, output_ghi)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_cloudiness, logits=outputs)
        return loss
