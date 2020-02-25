import pickle

import tensorflow as tf
from tensorflow.keras import losses

from src import logging
from src.data import preprocessing
from src.data.train import load_data
from src.model.base import Model

logger = logging.create_logger(__name__)


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


class Session(object):
    """Train a model in a supervised way.

    It assumes that the data is labeled (inputs, targets).
    """

    def __init__(
        self, model: Model, predict_ghi=True, batch_size=128, skip_non_cached=False,
    ):
        """Initialize a training session."""
        mse = losses.MeanSquaredError()

        def rmse(pred, target):
            """Wraper around TF MSE Loss."""
            return mse(pred, target) ** 0.5

        self.loss_fn = rmse
        self.model = model
        self.predict_ghi = predict_ghi
        self.batch_size = batch_size
        self.skip_non_cached = skip_non_cached

        self.scaling_ghi = preprocessing.min_max_scaling_ghi()

        self.metrics = {
            "train": tf.keras.metrics.Mean("train loss", dtype=tf.float32),
            "valid": tf.keras.metrics.Mean("valid loss", dtype=tf.float32),
            "test": tf.keras.metrics.Mean("test loss", dtype=tf.float32),
        }

        self.history = History()

    def train(
        self,
        optimizer: tf.keras.optimizers,
        epochs=25,
        enable_checkpoint=True,
        cache_file=None,
    ):
        """Performs the training of the model in minibatch."""
        config = self.model.config()
        logger.info(
            f"Starting training\n"
            + f" - Model: {self.model.title}\n"
            + f" - Config: {config}"
        )

        train_set, valid_set, _ = load_data(
            config=config, skip_non_cached=self.skip_non_cached,
        )

        logger.info("Apply Preprocessing")
        train_set = self.model.preprocess(train_set)
        valid_set = self.model.preprocess(valid_set)

        if cache_file is not None:
            train_set = train_set.cache(f"{cache_file}-train")
            valid_set = valid_set.cache(f"{cache_file}-valid")

        logger.info("Fitting model.")
        for epoch in range(epochs):
            logger.info("Training...")

            for i, data in enumerate(train_set.batch(self.batch_size)):
                inputs = data[:-1]
                targets = data[-1]

                self._train_step(optimizer, inputs, targets, i + 1)

            logger.info("Evaluating validation loss")
            self._evaluate("valid", valid_set, self.batch_size)

            logger.info("Checkpointing...")
            self.model.save(str(epoch))

            self._update_progress(epoch)
            self.history.save(f"{self.model.title}-{epoch}")

        logger.info("Done.")

    def test(self, checkpoint: str):
        """Test a trained model on the test set."""
        config = self.model.config()
        self.model.load(checkpoint)

        _, _, test_set = load_data(config=config, skip_non_cached=self.skip_non_cached)
        test_set = self.model.preprocess(test_set)

        self._evaluate("test", test_set, self.batch_size)
        self.history.save(f"{self.model.title}-{checkpoint}-test-set")

    def _update_progress(self, epoch):
        train_metric = self.metrics["train"]
        valid_metric = self.metrics["valid"]

        logger.info(
            f"Epoch: {epoch + 1}, Train loss: {train_metric.result()}, Valid loss: {valid_metric.result()} "
        )

        # Reset the cumulative metrics after each epoch
        self.history.record("train", train_metric.result())
        train_metric.reset_states()
        valid_metric.reset_states()

    def _evaluate(self, name, dataset, batch_size):
        metric = self.metrics[name]

        for i, data in enumerate(dataset.batch(batch_size)):
            inputs = data[:-1]
            targets = data[-1]

            loss = self._calculate_loss(inputs, targets)
            if self.predict_ghi:
                metric(self._rescale_loss_ghi(loss))
            else:
                metric(loss)

            logger.info(f"Batch [{i+1}]: {name}-set loss {metric.result()}")

        self.history.record(name, metric.result())

    def _train_step(self, optim, train_inputs, train_targets, batch):
        with tf.GradientTape() as tape:
            outputs = self.model(train_inputs, training=True)
            loss = self.loss_fn(train_targets, outputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optim.apply_gradients(zip(gradients, self.model.trainable_variables))

        metric = self.metrics["train"]
        if self.predict_ghi:
            metric(self._rescale_loss_ghi(loss))
        else:
            metric(loss)

        logger.info(f"Batch [{batch}]: train-set loss {metric.result()}")

    def _calculate_loss(self, valid_inputs, valid_targets):
        outputs = self.model(valid_inputs)
        return self.loss_fn(valid_targets, outputs)

    def _rescale_loss_ghi(self, loss):
        return self.scaling_ghi.original(loss) - self.scaling_ghi.min_value
