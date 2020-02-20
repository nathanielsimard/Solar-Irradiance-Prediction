from tensorflow.keras import losses, optimizers

from src.model import conv3d
from src.training import SupervisedTraining


def main():
    """Executable."""
    model = conv3d.CNN3D()
    model.load("5")
    optimizer = optimizers.Adam(0.001)
    loss_obj = losses.MeanSquaredError()

    def rmse(pred, target):
        return loss_obj(pred, target) ** 0.5

    training_session = SupervisedTraining(
        optimizer=optimizer, model=model, loss_fn=rmse
    )
    training_session.test_evaluation()


if __name__ == "__main__":
    main()
