import itertools
import logging
from typing import Iterator

from src.data import dataloader, split
from src.data.metadata import Coordinates, Metadata, MetadataLoader, Station

logger = logging.getLogger(__name__)

STATION_COORDINATES = {
    Station.BND: Coordinates(40.05192, -88.37309, 230),
    Station.TBL: Coordinates(40.12498, -105.23680, 1689),
    Station.DRA: Coordinates(36.62373, -116.01947, 1007),
    Station.FPK: Coordinates(48.30783, -105.10170, 634),
    Station.GWN: Coordinates(34.25470, -89.87290, 98),
    Station.PSU: Coordinates(40.72012, -77.93085, 376),
    Station.SXF: Coordinates(43.73403, -96.62328, 473),
}


TRAIN_CONFIG = dataloader.Config(
    error_strategy=dataloader.ErrorStrategy.skip,
    crop_size=(64, 64),
    features=[dataloader.Feature.target_ghi],
    channels=["ch1", "ch2", "ch3", "ch4", "ch6"],
)


def train(
    file_name="/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
    batch_size=64,
):
    logger.info("Start training")
    dataset_train, dataset_valid, dataset_test = _load_data(file_name)

    for i, (target,) in enumerate(dataset_train.batch(batch_size)):
        if i % 100 == 0:
            logger.info(f"Loader target with size {target}")


def _load_data(file_name):
    train_datetimes, valid_datetimes, test_datetimes = split.load()

    metadata_loader = MetadataLoader(file_name=file_name)
    metadata_train = metadata_station(metadata_loader, train_datetimes)
    metadata_valid = metadata_station(metadata_loader, valid_datetimes)
    metadata_test = metadata_station(metadata_loader, test_datetimes)

    dataset_train = dataloader.create_dataset(metadata_train, TRAIN_CONFIG)
    dataset_valid = dataloader.create_dataset(metadata_valid, TRAIN_CONFIG)
    dataset_test = dataloader.create_dataset(metadata_test, TRAIN_CONFIG)

    return dataset_train, dataset_valid, dataset_test


def metadata_station(metadata_loader, datetimes) -> Iterator[Metadata]:
    generators = []
    for station, coordinate in STATION_COORDINATES.items():
        generators.append(
            metadata_loader.load(
                station,
                coordinate,
                night_time=False,
                target_datetimes=datetimes,
                skip_missing=True,
            )
        )
    return itertools.chain(*generators)
