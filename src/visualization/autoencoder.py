from typing import List

import numpy as np
import tensorflow as tf
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

    _plt_images(originals, generateds, config.crop_size, config.channels)
    plt.savefig(f"assets/autoencoder.png")


def _plt_images(
    originals: List[np.ndarray],
    generated: List[np.ndarray],
    output_size,
    channels,
    scale=0.1,
):
    plt.cla()
    plt.clf()
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    num_col = 2
    num_rows = 2

    figsize_x = int(output_size[0] * scale)
    figsize_y = int(output_size[1] * scale)

    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_col, figsize=(figsize_x, figsize_y)
    )
    fig.tight_layout()
    for row in range(num_rows):
        original = originals[row]
        gen = generated[row]
        channel = channels[row]

        ax_original = axs[row][0]
        ax_gen = axs[row][1]

        ax_original.set_xticks([])
        ax_original.set_yticks([])
        ax_original.set_ylabel(channel)

        ax_gen.set_xticks([])
        ax_gen.set_yticks([])

        if row == 0:
            ax_gen.set_title("Generated")
            ax_original.set_title("Original")

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
