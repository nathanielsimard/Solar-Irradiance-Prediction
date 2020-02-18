from matplotlib import pyplot as plt

from src.data.train import load_data
from src.model.autoencoder import Autoencoder


def show_image():
    autoencoder = Autoencoder()
    autoencoder.load(str(24))

    config = autoencoder.config(training=False)

    train_dataset, _, _ = load_data(config=config)
    images = _first_image(train_dataset)
    for i, image in enumerate(images):
        plt.cla()
        plt.clf()
        plt.axis("off")
        plt.tight_layout()

        num_channels = image.shape[-1]
        for n in range(num_channels):
            image2d = image[:, :, n]
            plt.imshow(image2d, cmap="gray")
            name = f"assets/{n}-autoencoder.png"
            plt.savefig(name)


def _first_image(dataset, index=3):
    for i, (image, _) in enumerate(dataset.batch(1)):
        if i == index:
            return image
