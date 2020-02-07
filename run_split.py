import pickle

import pandas as pd

from src import logging
from src.data import split

logger = logging.create_logger(__name__)


def find_datetimes(
    df_file_name="/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
):
    logger.info("Finding datetimes")
    with open(df_file_name, "rb") as file:
        df: pd.Dataframe = pickle.load(file)
        timestamps = df.index.tolist()
        datetimes = [timestamp.to_pydatetime() for timestamp in timestamps]
        logger.info(f"Found {len(datetimes)} datetimes")

        return datetimes


def main():
    logger.info("Creating splits")
    datetimes = find_datetimes()
    train_set, valid_set, test_set = split.create_split(datetimes)
    logger.info("Slits created")
    split.persist_split(train_set, valid_set, test_set)
    logger.info("Slits persisted")


if __name__ == "__main__":
    main()
