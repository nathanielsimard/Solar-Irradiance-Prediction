import matplotlib.pyplot as plt

from src.session import History


def plot_learning_curve(file_name):
    """Plot the valid and train loss curves."""
    history = History.load(file_name)

    plt.cla()
    plt.clf()
    epochs = range(1, len(history.logs["train"]) + 1)

    plt.plot(epochs, history.logs["train"], label="train")
    plt.plot(epochs, history.logs["valid"], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE loss")
    plt.title("Learning curves")
    plt.legend()
    plt.savefig(file_name + ".png")
