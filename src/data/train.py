import itertools
from typing import Iterator, Tuple

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


def default_config():
    """Default training configurations."""
    return dataloader.Config(
        error_strategy=dataloader.ErrorStrategy.skip,
        crop_size=(64, 64),
        features=[dataloader.Feature.image, dataloader.Feature.target_ghi],
        channels=["ch1", "ch2", "ch3", "ch4", "ch6"],
    )


def load_data(
    file_name="/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
    batch_size=64,
    night_time=False,
    skip_missing=True,
    config=default_config(),
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, valid and test datasets.

    Return: (train_dataset, valid_dataset, test_dataset)
    """
    train_datetimes, valid_datetimes, test_datetimes = split.load()

    metadata_loader = MetadataLoader(file_name=file_name)
    metadata_train = metadata_station(
        metadata_loader,
        train_datetimes,
        night_time=night_time,
        skip_missing=skip_missing,
    )
    metadata_valid = metadata_station(
        metadata_loader,
        valid_datetimes,
        night_time=night_time,
        skip_missing=skip_missing,
    )
    metadata_test = metadata_station(
        metadata_loader,
        test_datetimes,
        night_time=night_time,
        skip_missing=skip_missing,
    )

    dataset_train = dataloader.create_dataset(metadata_train, config)
    dataset_valid = dataloader.create_dataset(metadata_valid, config)
    dataset_test = dataloader.create_dataset(metadata_test, config)

    logger.info("Loaded datasets.")
    return dataset_train, dataset_valid, dataset_test


def metadata_station(
    metadata_loader, datetimes, night_time=False, skip_missing=True
) -> Iterator[Metadata]:
    """Create metadata for all stations."""
    generators = []
    for station, coordinate in STATION_COORDINATES.items():
        generators.append(
            metadata_loader.load(
                station,
                coordinate,
                night_time=night_time,
                target_datetimes=datetimes,
                skip_missing=skip_missing,
            )
        )
    return itertools.chain(*generators)
