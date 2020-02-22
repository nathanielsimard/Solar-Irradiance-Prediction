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
    config.num_images = 6


    _, valid_dataset, _ = load_data(config=config)

    images_originals = _first_images(valid_dataset)
    images_originals = model.scaling_image.normalize(images_originals)

    image_pred = _predict_images(model, encoder, decoder, images_originals[:, :3])
    #images_originals = images_originals[:, 3:]

    images_originals = encoder(images_originals[0], training=False)
    images_originals = decoder(images_originals, training=False)

    images_originals = model.scaling_image.original(images_originals)
    image_pred = model.scaling_image.original(np.array(image_pred))

    generateds = []
    originals = []

    for i in range(3):
        generated = image_pred[i]
        print(images_originals.shape)
        original = images_originals[i]
        channel = 0

        originals.append(original[:, :, channel])
        generateds.append(generated[:, :, channel])

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


def _predict_images(model: LanguageModel,encoder: Encoder, decoder: Decoder, images_features):
    flatten = tf.keras.layers.Flatten()
    preds = []
    inputs = encoder(images_features[0], training=False)
    inputs = tf.expand_dims(inputs,0)
    num_generate = 3

    for i in range(num_generate):
        pred_features = model((inputs,), training=False)
        inputs = tf.concat([inputs[:, :], pred_features[:, -1:]], 1)

        pred_features = tf.squeeze(pred_features, 0)
        # Last batch dim
        next_feature = pred_features[-1:]
        pred_images = decoder((next_feature), False)
        # Remove the batch dim
        preds.append(pred_images[0])

    return preds


def _first_images(dataset, index=3):
    a = 0
    for data in dataset.batch(1):
        if data[0].shape != (1, 6, 64, 64, 5):
            continue
        a += 1
        if a == 100:
            return data[0]
