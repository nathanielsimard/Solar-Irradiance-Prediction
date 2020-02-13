import os
from typing import List

from matplotlib import pyplot as plt

from src.data.dataloader import Feature
from src.data.train import default_config, load_data


def plot3d():
    """Create a GIF from 3D image in the training set."""
    for channel in ["ch1", "ch2", "ch3", "ch4", "ch6"]:
        config = default_config()
        config.features = [Feature.image]
        config.channels = [channel]
        config.crop_size = (300, 300)
        config.time_interval_min = 30
        config.num_images = 10

        train_dataset, _, _ = load_data(config=config)
        images = _first_image(train_dataset)

        image_names = []
        for i, image in enumerate(images[0]):
            plt.cla()
            plt.clf()
            plt.axis("off")
            plt.tight_layout()

            image2d = image[:, :, 0]
            plt.imshow(image2d, cmap="gray")

            name = f"assets/{i}.png"
            plt.savefig(name)

            image_names.append(name)
        make_gif("assets/image-3d-{channel}.gif", image_names)


def make_gif(file_name: str, image_names: List[str]):
    """Make a gif from the image_paths.

    'convert' must be installed on the computer.
    It is normaly available on linux machines.
    """
    images_str = ""
    for name in image_names:
        images_str += f" {name}"

    cmd_create = f"convert {images_str} {file_name}"
    cmd_remove = f"rm {images_str}"

    os.system(cmd_create)
    os.system(cmd_remove)


def _first_image(dataset, index=3):
    for i, (image,) in enumerate(dataset.batch(1)):
        if i == index:
            return image
