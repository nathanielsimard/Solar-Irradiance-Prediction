from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.data.train import load_data
from src.model.autoencoder import Decoder, Encoder
from src.model.seq2seq import Gru


def plot_comparison(encoder_instance: str, seq2seq_instance: str):
    """Show original and generated futur images in a grid."""
    encoder = Encoder()
    encoder.load(encoder_instance)
    model = Gru(encoder)
    model.load(seq2seq_instance)
    config = model.config()
    config.num_images = 6
    config.skip_missing_past_images = True

    decoder = Decoder(len(config.channels))
    decoder.load(encoder_instance)

    valid_dataset, _, _ = load_data(config=config)
    images_originals = _first_images(valid_dataset, 6)

    image_pred = model.predict_next_images(images_originals[0, :3], 3)

    before = model.scaling_image.normalize(images_originals[0, :3])
    before = encoder(before, training=False)
    before = decoder(before, training=False)

    images_originals = model.scaling_image.normalize(images_originals[0, 3:])
    images_originals = encoder(images_originals, training=False)
    images_originals = decoder(images_originals, training=False)

    before = model.scaling_image.original(before)
    images_originals = model.scaling_image.original(images_originals)
    image_pred = model.scaling_image.original(np.array(image_pred))

    generateds = []
    originals = []
    befores = []

    for i in range(3):
        generated = image_pred[i]
        original = images_originals[i]
        bef = before[i]
        channel = 0

        originals.append(original[:, :, channel])
        generateds.append(generated[:, :, channel])
        befores.append(bef[:, :, channel])

    _plt_images(originals, generateds, befores, config.crop_size)
    plt.savefig(f"assets/seq2seq.png")


def _plt_images(
    originals: List[np.ndarray],
    generated: List[np.ndarray],
    befores,
    output_size,
    scale=0.1,
):
    plt.cla()
    plt.clf()

    num_rows = len(originals)
    num_col = 3

    figsize_x = int(output_size[0] * scale)
    figsize_y = int(output_size[1] * scale)

    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_col, figsize=(figsize_x, figsize_y)
    )
    fig.tight_layout()
    for row, (original, gen, bef) in enumerate(zip(originals, generated, befores)):
        ax_bef = axs[row][0]
        ax_original = axs[row][1]
        ax_gen = axs[row][2]

        ax_bef.set_xticks([])
        ax_bef.set_yticks([])

        ax_original.set_xticks([])
        ax_original.set_yticks([])

        ax_gen.set_xticks([])
        ax_gen.set_yticks([])

        if row == 0:
            ax_bef.set_title("Previous")
            ax_original.set_title("Future")
            ax_gen.set_title("Predicted")

        ax_bef.imshow(bef, cmap="gray")
        ax_original.imshow(original, cmap="gray")
        ax_gen.imshow(gen, cmap="gray")


def _first_images(dataset, num, index=3):
    a = 0
    for data in dataset.batch(1):
        if data[0].shape != (1, num, 64, 64, 5):
            continue
        a += 1
        if a == index:
            return data[0]
