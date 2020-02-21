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
    encoder.load("3")
    model = LanguageModel(encoder)
    model.load("22")

    decoder = Decoder(num_channels)
    decoder.load(encoder_instance)

    config = model.config(training=False)

    _, valid_dataset, _ = load_data(config=config)
    dataset = model.preprocess(valid_dataset)

    images = _first_images(dataset)
    images_originals = _first_images(valid_dataset)
    image_pred = _predict_images(model, decoder, images)

    generateds = []
    originals = []

    for i in range(6):
        generated = image_pred[i]
        original = images_originals[0, i]

        num_channels = original.shape[-1]
        originals.append(original[:, :, 1])
        generateds.append(generated[:, :, 1])

    _plt_images(originals, generateds, config.crop_size)
    plt.savefig(f"assets/languagemodel.png")


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
    inputs = images_features
    num_generate = 6

    for i in range(num_generate):
        pred_features = model(inputs, False)
        inputs = tf.concat([inputs[:, :], pred_features[:, -1:]], 0)

        pred_features = tf.squeeze(pred_features, 0)
        next_feature = pred_features[-1]
        # Remove the batch dim
        next_feature = tf.reshape(next_feature, (1, 8, 8, 32))
        pred_images = decoder((next_feature), False)
        pred = model.scaling_image.original(pred_images)
        preds.append(pred[0])

    return preds


def _first_images(dataset, index=3):
    for i, data in enumerate(dataset.batch(1)):
        if i == index:
            return data[0]
