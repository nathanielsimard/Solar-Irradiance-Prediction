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

    image_tt = autoencoder.scaling_image.normalize(images)
    print(image_tt.shape)
    images_pred = autoencoder((image_tt), False)
    print(images_pred.shape)
    images_pred = autoencoder.scaling_image.original(images_pred)
    print(images_pred.shape)


    for i, (image, image_pred) in enumerate(zip(images, images_pred)):
        num_channels = image.shape[-1]
        for n in range(num_channels):
            plt.cla()
            plt.clf()
            plt.axis("off")
            plt.tight_layout()
            image2d = image[:, :, n]
            plt.imshow(image2d, cmap="gray")
            name = f"assets/{n}-autoencoder.png"
            plt.savefig(name)

            plt.cla()
            plt.clf()
            plt.axis("off")
            plt.tight_layout()
            image2d_pred = image_pred[:, :, n]
            plt.imshow(image2d_pred, cmap="gray")
            name = f"assets/{n}-autoencoder-pred.png"
            plt.savefig(name)

def _first_image(dataset, index=3):
    for i, data in enumerate(dataset.batch(1)):
        if i == index:
            return data[0]
