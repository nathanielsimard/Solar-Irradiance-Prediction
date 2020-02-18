from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.data.train import load_data
from src.model.autoencoder import Autoencoder


def show_images(
    originals: List[np.ndarray], generated: List[np.ndarray], output_size, scale=0.1
):
    """Show images in a grid."""
    plt.cla()
    plt.clf()

    num_rows = len(originals)
    num_col = 2

    figsize_x = int(output_size[0] * scale)
    figsize_y = int(output_size[1] * scale)

    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_col, figsize=(figsize_x, figsize_y)
    )
    for row, (original, gen) in enumerate(zip(originals, generated)):
        ax_original = axs[row][0]
        ax_gen = axs[row][1]

        ax_original.set_xticks([])
        ax_original.set_yticks([])

        ax_gen.set_xticks([])
        ax_gen.set_yticks([])

        ax_original.imshow(original, cmap="gray")
        ax_gen.imshow(gen, cmap="gray")


def show_image():
    autoencoder = Autoencoder()
    autoencoder.load(str(24))

    config = autoencoder.config(training=False)

    train_dataset, _, _ = load_data(config=config)
    images = _first_image(train_dataset)
    images_pred = _predict_images(autoencoder, images)

    images_preds = []
    images_originals = []
    for i, (image, image_pred) in enumerate(zip(images, images_pred)):
        num_channels = image.shape[-1]
        for n in range(num_channels):
            images_originals.append(image[:, :, n])
            images_preds.append(image_pred[:, :, n])

    show_images(images_originals, images_preds, config.crop_size)
    plt.savefig(f"assets/autoencoder.png")


def _predict_images(autoencoder, images):
    images_scaled = autoencoder.scaling_image.normalize(images)
    images_scaled_pred = autoencoder((images_scaled), False)
    return autoencoder.scaling_image.original(images_scaled_pred)


def _first_image(dataset, index=3):
    for i, data in enumerate(dataset.batch(1)):
        if i == index:
            return data[0]
