from matplotlib import pyplot as plt

from src import logging
from src.data.dataloader import Feature
from src.data.train import default_config, load_data

logger = logging.create_logger(__name__)


def plot3d(channel="ch1"):
    config = default_config()
    config.features = [Feature.image]
    config.channels = [channel]
    config.crop_size = (300, 300)
    config.time_interval_min = 60
    config.num_images = 7

    train_dataset, _, _ = load_data(config=config)

    images = _first_image(train_dataset)

    logger.info(images.shape)
    for i, img in enumerate(images[0]):
        logger.info(img.shape)

        plt.cla()
        plt.clf()
        plt.axis("off")
        imgg = img[:, :, 0]

        logger.info(imgg.shape)
        plt.imshow(imgg, cmap="gray")
        plt.savefig(f"image-3D/{i}.png")


def _first_image(dataset):
    for (image,) in dataset.batch(1):
        return image
