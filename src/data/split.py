import os
import pickle
import random
from datetime import datetime
from typing import Iterator, List, Tuple

from src import env

TRAIN_SET_FILE_NAME = "train_set.pkl"
VALID_SET_FILE_NAME = "valid_set.pkl"
TEST_SET_FILE_NAME = "test_set.pkl"


def create_split(
    datetimes: Iterator[datetime], valid_ratio=0.2
) -> Tuple[List[datetime], List[datetime], List[datetime]]:
    """Create Train, Validation and Test set from datetimes.

    All datimes from 2015 will be in the test set.
    The remaining datetimes will be shuffled and split with the valid_ratio.
    """
    test_set = [date for date in datetimes if date >= datetime(2015, 1, 1)]
    valid_set = [
        date
        for date in datetimes
        if date >= datetime(2014, 1, 1) and date < datetime(2015, 1, 1)
    ]
    train_set = [date for date in datetimes if date < datetime(2014, 1, 1)]
    random.shuffle(train_set)

    return (train_set, valid_set, test_set)


def persist_split(
    train_set: List[datetime],
    valid_set: List[datetime],
    test_set: List[datetime],
    dir_path: str = None,
) -> None:
    """Save the split in a pickle to always reuse the same split and avoid errors."""
    if dir_path is None:
        dir_path = env.get_split_path()
    os.makedirs(dir_path, exist_ok=True)
    _persist(dir_path, TRAIN_SET_FILE_NAME, train_set)
    _persist(dir_path, VALID_SET_FILE_NAME, valid_set)
    _persist(dir_path, TEST_SET_FILE_NAME, test_set)


def load(dir_path=None):
    """Load the persisted train, valid and test set."""
    if dir_path is None:
        dir_path = env.get_split_path()
    train_set = _load(dir_path, TRAIN_SET_FILE_NAME)
    valid_set = _load(dir_path, VALID_SET_FILE_NAME)
    test_set = _load(dir_path, TEST_SET_FILE_NAME)

    return (train_set, valid_set, test_set)


def _persist(dir_path, file_name, obj):
    with open(f"{dir_path}/{file_name}", "wb") as file:
        pickle.dump(obj, file)


def _load(dir_path, file_name):
    with open(f"{dir_path}/{file_name}", "rb") as file:
        return pickle.load(file)
