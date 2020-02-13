from matplotlib import pyplot as plt

from src.data.dataloader import Feature
from src.data.train import default_config, load_data


def plot3d(image_path, offset, channel="ch1"):
    config = default_config()
    config.features = [Feature.image]

    dataset = load_data(config=config)
    i = 0
    for (image,) in dataset.batch(1):
        i += 1
        if i > 6:
            images = image
            break

    for i, img in enumerate(images[0]):
        plt.cla()
        plt.clf()
        plt.imshow(img, cmap="gray")
        plt.savefig(f"image-3D-{i}.png")
