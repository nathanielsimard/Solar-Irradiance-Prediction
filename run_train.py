from src.model import conv2d
from src.data import train
from tensorflow.keras import optimizers, losses


def main():
    """Executable."""
    model = conv2d.CNN2D()
    optimizer = optimizers.SGD(0.0001)
    loss_obj = losses.MeanSquaredError()
    training_session = train.Training(
        optimizer=optimizer, model=model, loss_fn=loss_obj
    )
    training_session.training()


if __name__ == "__main__":
    main()
