import argparse
import os

from src import logging
from src.data.dataloader import Feature
from src.data.train import default_config, load_data

logger = logging.create_logger(__name__)


BATCH_SIZE = 128


def cache(size, cache_dir):
    config = default_config()
    config.features = [Feature.image]
    config.crop_size = size

    # Create image cache dir
    config.image_cache_dir = cache_dir + f"/image_cache_{size}"
    config.image_cache_dir = config.image_cache_dir.replace("(", "")
    config.image_cache_dir = config.image_cache_dir.replace(",", "")
    config.image_cache_dir = config.image_cache_dir.replace(")", "")
    config.image_cache_dir = config.image_cache_dir.replace(" ", "-")

    logger.info(f"Caching images with size {size} in dir {config.image_cache_dir}")

    dataset_train, dataset_valid, dataset_test = load_data(
        enable_tf_caching=False, config=config
    )

    _create_cache("train", dataset_train)
    _create_cache("valid", dataset_valid)
    _create_cache("test", dataset_test)

    os.system(f"tar -cf {config.image_cache_dir}.tar {config.image_cache_dir}")


def _create_cache(name, dataset):
    for i, _ in enumerate(dataset.batch(BATCH_SIZE)):
        logger.info(f"Cached {i * BATCH_SIZE} {name} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size-x", help="Image size in x direction", type=int, required=True,
    )
    parser.add_argument(
        "--size-y", help="Image size in y direction", type=int, required=True,
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory where image are going to be cache and compressed",
        type=str,
        default="/project/cq-training-1/project1/teams/team10",
    )
    args = parser.parse_args()

    size = (args.size_x, args.size_y)
    cache(size, args.cache_dir)
