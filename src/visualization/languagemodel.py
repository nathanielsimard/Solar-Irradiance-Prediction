from typing import List

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.data.train import load_data
from src.model.autoencoder import Decoder, Encoder
from src.model.languagemodel import LanguageModel


def plot_comparison(
    encoder_instance: str, language_model_instance: str, num_channels=5
):
    """Show original and generated futur images in a grid."""
    encoder = Encoder()
    encoder.load(encoder_instance)
    model = LanguageModel(encoder)
    model.load(language_model_instance)

    decoder = Decoder(num_channels)
    decoder.load(encoder_instance)

    config = model.config(training=False)

    _, valid_dataset, _ = load_data(config=config)
    dataset = model.preprocess(valid_dataset)

    images = _first_images(dataset)
    image_pred = _predict_images(model, decoder, images)

    generateds = []
    originals = []

    for i, (original, generated) in enumerate(zip(images, image_pred)):
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


def _predict_images(model: LanguageModel, decoder: Decoder, images_features):
    preds = []
    inputs = tf.expand_dims(images_features, 0)
    model.reset_states()
    num_generate = len(images_features)

    for i in range(num_generate):
        pred_features, _ = model(inputs, False)
        inputs = pred_features

        # Remove the batch dim
        pred_features = tf.squeeze(pred_features, 0)
        pred_images = decoder((pred_features), False)
        pred = model.scaling_image.original(pred_images)
        preds.append(pred)

    return preds


def _first_images(dataset, index=3):
    for i, data in enumerate(dataset.batch(1)):
        if i == index:
            return data[0]
