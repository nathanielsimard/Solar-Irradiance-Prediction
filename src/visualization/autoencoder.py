from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.data.train import load_data
from src.model.autoencoder import Autoencoder


def show_images(images: List[np.ndarray], output_size, scale=0.1):
    """Show images in a grid."""
    plt.cla()
    plt.clf()

    figsize_x = int(output_size[0] * scale)
    figsize_y = int(output_size[1] * scale)

    fig, axs = plt.subplots(nrows=len(images), figsize=(figsize_x, figsize_y))
    for row, image in enumerate(images):
        ax = axs[row]
        image = images[row]

        if row == 0:
            ax.set_title("img")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(image, cmap="gray")


def show_image():
    autoencoder = Autoencoder()
    autoencoder.load(str(24))

    config = autoencoder.config(training=False)

    train_dataset, _, _ = load_data(config=config)
    images = _first_image(train_dataset)
    imgs = []
    for i, image in enumerate(images):
        plt.cla()
        plt.clf()
        plt.axis("off")
        plt.tight_layout()

        num_channels = image.shape[-1]
        for n in range(num_channels):
            image2d = image[:, :, n]
            imgs.append(image2d)
            plt.imshow(image2d, cmap="gray")
            name = f"assets/{n}-autoencoder.png"
            plt.savefig(name)
    show_images([imgs], config.crop_size)


def _first_image(dataset, index=3):
    for i, (image, _) in enumerate(dataset.batch(1)):
        if i == index:
            return image
