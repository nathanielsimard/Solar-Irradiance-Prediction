from typing import List
import tensorflow as tf

import numpy as np
from matplotlib import pyplot as plt

from src.data.train import load_data
from src.model.autoencoder import Autoencoder


def plot_comparison(instance: str):
    """Show original and generated images in a grid."""
    autoencoder = Autoencoder()
    autoencoder.load(instance)

    config = autoencoder.config(training=False)

    _, valid_dataset, _ = load_data(config=config)
    image = _first_image(valid_dataset)
    image_pred = _predict_image(autoencoder, image)

    generateds = []
    originals = []

    for i, (original, generated) in enumerate(zip(image, image_pred)):
        num_channels = original.shape[-1]
        for n in range(num_channels):
            originals.append(original[:, :, n])
            generateds.append(generated[:, :, n])

    _plt_images(originals, generateds, config.crop_size)
    plt.savefig(f"assets/autoencoder.png")


def _plt_images(
    originals: List[np.ndarray], generated: List[np.ndarray], output_size, scale=0.1
):
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


def _predict_image(autoencoder, images):
    images_scaled = autoencoder.scaling_image.normalize(images)
    images_scaled = tf.expand_dims(images_scaled, 0)
    images_scaled_pred = autoencoder((images_scaled), False)
    return autoencoder.scaling_image.original(images_scaled_pred)


def _first_image(dataset, index=2):
    for i, data in enumerate(dataset.batch(1)):
        if i == index:
            return data[0]
