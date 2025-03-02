import itertools
import random
from typing import Callable, Iterator, Tuple

import tensorflow as tf

import src.env as env
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
        return env.get_image_reader_cache_directory()
    except KeyError:
        return "/tmp"


def default_config():
    """Default training configurations."""
    return dataloader.DataloaderConfig(
        error_strategy=dataloader.ErrorStrategy.skip,
        crop_size=(64, 64),
        image_cache_dir=default_cache_dir(),
        features=[dataloader.Feature.image, dataloader.Feature.target_ghi],
        channels=["ch1", "ch2", "ch3", "ch4", "ch6"],
    )


def load_data(
    file_name=None,
    batch_size=64,
    night_time=False,
    skip_missing=True,
    config=default_config(),
    skip_non_cached=False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, valid and test datasets.

    Return: (train_dataset, valid_dataset, test_dataset)
    """
    if file_name is None:
        file_name = env.get_catalog_path()
    if env.run_local:
        config.local_path = env.get_local_data_path() + "/hdf5v7_8bit"

    # Both concepts are equivalent. If we force caching, we need to skip non cached images.
    config.force_caching = skip_non_cached

    train_datetimes, valid_datetimes, test_datetimes = split.load()

    random.shuffle(train_datetimes)
    random.shuffle(valid_datetimes)
    random.shuffle(test_datetimes)

    ratio_train_datetimes = int(len(train_datetimes) * config.ratio)
    ratio_valid_datetimes = int(len(valid_datetimes) * config.ratio)
    ratio_test_datetimes = int(len(test_datetimes) * config.ratio)

    logger.info(f"Loading {config.ratio*100}% of the data")
    logger.info(f"Training dataset has {ratio_train_datetimes} datetimes")
    logger.info(f"Validation dataset has {ratio_valid_datetimes} datetimes")
    logger.info(f"Test dataset has {ratio_test_datetimes} datetimes")
    logger.info(f"Using {len(STATION_COORDINATES)} stations")

    train_datetimes = train_datetimes[:ratio_train_datetimes]
    valid_datetimes = valid_datetimes[:ratio_valid_datetimes]
    test_datetimes = test_datetimes[:ratio_test_datetimes]

    if dataloader.Feature.metadata in config.features:
        config.precompute_clearsky = True
        target_datetimes = train_datetimes + valid_datetimes + test_datetimes
        config.target_datetimes = target_datetimes
        config.stations = STATION_COORDINATES

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

    dataset_train = dataloader.create_dataset(
        metadata_train, config, train_datetimes, STATION_COORDINATES
    )
    dataset_valid = dataloader.create_dataset(
        metadata_valid, config, valid_datetimes, STATION_COORDINATES
    )
    dataset_test = dataloader.create_dataset(
        metadata_test, config, test_datetimes, STATION_COORDINATES
    )

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
