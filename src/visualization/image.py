from matplotlib import pyplot as plt

import os
from src.data.dataloader import Feature
from src.data.train import default_config, load_data


def plot3d(channel="ch6"):
    config = default_config()
    config.features = [Feature.image]
    config.channels = [channel]
    config.crop_size = (300, 300)
    config.time_interval_min = 30
    config.num_images = 10

    train_dataset, _, _ = load_data(config=config)
    images = _first_image(train_dataset)

    images_name = []
    for i, image in enumerate(images[0]):
        plt.cla()
        plt.clf()
        plt.axis("off")
        image2d = image[:, :, 0]

        plt.imshow(image2d, cmap="gray")
        name = f"assets/{i}.png"
        images_name.append(name)
        plt.savefig(name)

    _make_gif("assets/image-6.gif", images_name)

def _make_gif(file_name, images):
    cmd = ""
    for image in images:
        cmd += f" {image}"

    cmd_create = f"convert {cmd} {file_name}"
    cmd_remove = f"rm {cmd}"

    os.system(cmd_create)
    os.system(cmd_remove)


def _first_image(dataset):
    for (image,) in dataset.batch(1):
        return image
