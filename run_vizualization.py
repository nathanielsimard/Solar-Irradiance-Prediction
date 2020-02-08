from datetime import datetime
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.data import dataloader
from src.data.metadata import Coordinates, Metadata

BND_COORDINATES = Coordinates(40.05192, -88.37309, 230)
TBL_COORDINATES = Coordinates(40.12498, -105.23680, 1689)
DRA_COORDINATES = Coordinates(36.62373, -116.01947, 1007)
FPK_COORDINATES = Coordinates(48.30783, -105.10170, 634)
GWN_COORDINATES = Coordinates(34.25470, -89.87290, 98)
PSU_COORDINATES = Coordinates(40.72012, -77.93085, 376)
SXF_COORDINATES = Coordinates(43.73403, -96.62328, 47)

IMAGE_PATH_1 = "tests/data/samples/2015.11.01.0800.h5"
IMAGE_PATH_2 = "tests/data/samples/2015.11.02.0800.h5"


def create_dataset(image_paths, channels, output_size, offsets, coordinates):
    def metadata():
        return _metadata_iterable(image_paths, offsets, coordinates)

    config = dataloader.Config(
        crop_size=output_size,
        channels=channels,
        features=[dataloader.Feature.image],
        image_cache_dir="/tmp",
    )

    return dataloader.create_dataset(metadata, config=config)


def _metadata_iterable(image_paths, offsets, coordinates):
    for image_path in image_paths:
        for offset in offsets:
            yield Metadata(
                image_path,
                "8bits",
                offset,
                datetime.now(),
                coordinates,
                target_ghi=100,
                target_ghi_1h=100,
                target_ghi_3h=100,
                target_ghi_6h=100,
            )


def show_images(
    images: List[List[np.ndarray]], channels, offsets, output_size, scale=0.1
):
    """Show images in a grid."""
    plt.cla()
    plt.clf()

    num_rows = len(offsets)
    num_col = len(channels)

    figsize_x = int(output_size[0] * scale)
    figsize_y = int(output_size[1] * scale)

    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_col, figsize=(figsize_x, figsize_y)
    )
    for row, offset in enumerate(offsets):
        for col, channel in enumerate(channels):
            ax = axs[row][col]
            image = images[row][col]

            if row == 0:
                ax.set_title(channel)
            if col == 0:
                ax.set_ylabel(offset)

            ax.set_xticks([])
            ax.set_yticks([])

            ax.imshow(image, cmap="gray")


def flatten_channels(image_channels: np.ndarray) -> List[np.ndarray]:
    """Flatten the images.

    Transform an image from a tensor [height, width, channel]
    to a list of 2D images.
    """
    num_channels = image_channels.shape[2]
    image_shape = image_channels.shape[0:2]
    output = np.empty([num_channels] + list(image_shape))

    for i in range(num_channels):
        output[i] = image_channels[:, :, i]
        output[i] = output[i].astype(np.uint8)
    return output


def visualize():
    offsets = list(range(2))
    channels = ["ch1", "ch2", "ch3", "ch4", "ch6"]
    output_size = (300, 300)
    for name, coordinates in [
        ("BND", BND_COORDINATES),
        ("TBL", TBL_COORDINATES),
        ("DRA", DRA_COORDINATES),
        ("FPK", FPK_COORDINATES),
        ("GWN", GWN_COORDINATES),
        ("PSU", PSU_COORDINATES),
        ("SXF", SXF_COORDINATES),
    ]:
        images = [
            image
            for (image,) in create_dataset(
                [IMAGE_PATH_1], channels, output_size, offsets, coordinates
            )
        ]

        images = [flatten_channels(image) for image in images]
        show_images(images, channels, offsets, output_size)
        plt.savefig(f"{name}.png")


if __name__ == "__main__":
    visualize()
