import tensorflow as tf
from tf.keras import Model
from tk.keras import optimizers, losses, metrics

CHECKPOINT_TIMESTAMP = 5


def train(
    train_dataset: tf.data.Dataset, model: Model, optimizer: optimizers,
):
    """Performs the training over a specified number of epochs.

    TODO: revisit all the train.py with a clearer head.
    """
    train_loss = metrics.Mean(name="train_loss")

    # TODO: adapt the training loops with a batch_size parameter
    for inputs, targets in train_dataset:
        preds, loss = _train_step(model, optimizer, inputs, targets, training=True)
        train_loss.update_state(loss)

    if tf.equal(optimizer.iteration() % CHECKPOINT_TIMESTAMP, 0):
        model.save(
            "/project/cq-training-1/project1/teams/team10/checkpoints/name_of_my_model.h5"
        )
        tf.summary.scalar("loss", train_loss.result(), step=optimizer.iterations)
        train_loss.reset_states()
        # TODO: precise the save directory for the checkpoints and the results.
        # TODO: incorporate the tf.summary to write up our results instead of using pickle.
        # To use tensorboard: tensorboard --logdir /path/to/summaries
        # see effective tensorflow 2.0 @ https://www.tensorflow.org/guide/effective_tf2


def test(valid_dataset: tf.data.Dataset, model: Model, num_epoch: int):
    """TODO: revisit all the train.py with a clearer head."""

    valid_loss = metrics.Mean(name="valid_loss")
    for inputs, targets in valid_dataset:
        v_preds, v_loss = _validation_step(model, inputs, targets, training=False)
        valid_loss.update_state(v_loss)
    tf.summary.scalar("validation_loss", valid_loss.result(), step=num_epoch)
    valid_loss.reset_states()


@tf.function
def _train_step(
    model: Model, optimizer: optimizers, inputs, targets, training: bool,
):
    """Performs one epochs of training."""
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = losses.MSE(targets, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return (predictions, loss)


@tf.function
def _validation_step(model: Model, inputs, targets, training: bool):
    """Performs a forward pass to evaluate the validation loss."""
    predictions = model(inputs, training=False)
    validation_loss = losses.MSE(targets, predictions)
    return (predictions, validation_loss)


def main():
    train_summary_writer = tf.summary.create_file_writer(
        "/project/cq-training-1/project1/teams/team10/summaries/train"
    )
    valid_summary_writer = tf.summary.create_file_writer(
        "/project/cq-training-1/project1/teams/team10/summaries/valid"
    )
    model = 0
    optimizer = 0
    train_dataset = 0
    valid_dataset = 0
    with train_summary_writer.as_default():
        train(model, optimizer, train_dataset)
    with valid_summary_writer.as_default():
        test(valid_dataset, model, num_epoch=optimizer.iterations)

    pass


if __name__ == "__main__":
    main()
