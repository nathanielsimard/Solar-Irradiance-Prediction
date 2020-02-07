from src.data import train, dataloader
from src import logging
import numpy as np

logger = logging.create_logger(__name__)


def reduce_max(acc, x):
    max_x = np.max(x.numpy())
    return acc if acc > max_x else max_x


def reduce_min(acc, x):
    min_x = np.min(x.numpy())
    return acc if acc < min_x else min_x


def main():
    config = train.default_config()
    config.features = [dataloader.Feature.target_ghi]

    data_train, _, _ = train.load_data(config=config)

    max_target = data_train.reduce(0, reduce_max)
    min_target = data_train.reduce(max_target, reduce_min)

    logger.info(f"max target value: {max_target}")
    logger.info(f"min target value: {min_target}")


if __name__ == "__main__":
    main()