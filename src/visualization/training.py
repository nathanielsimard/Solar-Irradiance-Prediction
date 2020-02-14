from src.training import History
import matplotlib.pyplot as plt


def plot_learning_curve(file_name):
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
