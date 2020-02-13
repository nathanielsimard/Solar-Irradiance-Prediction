from tensorflow.keras import losses, optimizers

from src.model import conv2d
from src.training import SupervisedTraining


def main():
    """Executable."""
    model = conv2d.CNN2D()
    optimizer = optimizers.SGD(0.0001)
    loss_obj = losses.MeanSquaredError()
    training_session = SupervisedTraining(
        optimizer=optimizer, model=model, loss_fn=loss_obj
    )
    training_session.run()


if __name__ == "__main__":
    main()
