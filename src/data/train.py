import itertools
import os
from typing import Callable, Iterator, Tuple

import tensorflow as tf

from src import logging
from src.data import dataloader, split
from src.data.metadata import Coordinates, Metadata, MetadataLoader, Station

logger = logging.create_logger(__name__)

STATION_COORDINATES = {
    Station.BND: Coordinates(40.05192, -88.37309, 230),
    Station.TBL: Coordinates(40.12498, -105.23680, 1689),
    Station.DRA: Coordinates(36.62373, -116.01947, 1007),
    Station.FPK: Coordinates(48.30783, -105.10170, 634),
    Station.GWN: Coordinates(34.25470, -89.87290, 98),
    Station.PSU: Coordinates(40.72012, -77.93085, 376),
    Station.SXF: Coordinates(43.73403, -96.62328, 473),
}


def default_cache_dir():
    """Use SCRATCH directory on helios, tmp otherwise."""
    try:
        return os.environ["SCRATCH"]
    except KeyError:
        return "/tmp"


def default_config():
    """Default training configurations."""
    return dataloader.Config(
        error_strategy=dataloader.ErrorStrategy.skip,
        crop_size=(64, 64),
        image_cache_dir=default_cache_dir(),
        features=[dataloader.Feature.image, dataloader.Feature.target_ghi],
        channels=["ch1", "ch2", "ch3", "ch4", "ch6"],
    )


def load_data(
    file_name="/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
    night_time=False,
    skip_missing=True,
    config=default_config(),
    enable_tf_caching=False,
    cache_file=default_cache_dir(),
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, valid and test datasets.

    Return: (train_dataset, valid_dataset, test_dataset)
    """
    train_datetimes, valid_datetimes, test_datetimes = split.load()
    ratio_train_datetimes = int(len(train_datetimes) * config.ratio)
    ratio_valid_datetimes = int(len(train_datetimes) * config.ratio)

    train_datetimes = train_datetimes[:ratio_train_datetimes]
    valid_datetimes = valid_datetimes[:ratio_valid_datetimes]

    metadata_loader = MetadataLoader(file_name=file_name)
    metadata_train = metadata_station(
        metadata_loader,
        train_datetimes,
        config.num_images,
        config.time_interval_min,
        night_time=night_time,
        skip_missing=skip_missing,
    )
    metadata_valid = metadata_station(
        metadata_loader,
        valid_datetimes,
        config.num_images,
        config.time_interval_min,
        night_time=night_time,
        skip_missing=skip_missing,
    )
    metadata_test = metadata_station(
        metadata_loader,
        test_datetimes,
        config.num_images,
        config.time_interval_min,
        night_time=night_time,
        skip_missing=skip_missing,
    )

    dataset_train = dataloader.create_dataset(metadata_train, config)
    dataset_valid = dataloader.create_dataset(metadata_valid, config)
    dataset_test = dataloader.create_dataset(metadata_test, config)

    if enable_tf_caching:
        dataset_train = dataset_train.cache(cache_file + "_train")
        dataset_test = dataset_test.cache(cache_file + "_test")
        dataset_valid = dataset_valid.cache(cache_file + "_valid")

    logger.info("Loaded datasets.")
    return dataset_train, dataset_valid, dataset_test


def metadata_station(
    metadata_loader,
    datetimes,
    num_images,
    time_interval_min,
    night_time=False,
    skip_missing=True,
) -> Callable[[], Iterator[Metadata]]:
    """Create metadata for all stations."""

    def gen():
        generators = []
        for station, coordinate in STATION_COORDINATES.items():
            generators.append(
                metadata_loader.load(
                    station,
                    coordinate,
                    night_time=night_time,
                    target_datetimes=datetimes,
                    skip_missing=skip_missing,
                    num_images=num_images,
                    time_interval_min=time_interval_min,
                )
            )
        return itertools.chain(*generators)

    return gen
