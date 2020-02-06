import logging
import pickle

import pandas as pd

from src.data import split


def find_datetimes(
    df_file_name="/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
):
    logging.info("Finding datetimes")
    with open(df_file_name, "rb") as file:
        df: pd.Dataframe = pickle.load(file)
        datetimes = df.index.tolist()
        logging.info(f"Found {len(datetimes)} datetimes")

        return datetimes


def main():
    logging.info("Creating splits")
    datetimes = find_datetimes()
    train_set, valid_set, test_set = split.create_split(datetimes)
    logging.info("Slits created")
    split.persist_split(train_set, valid_set, test_set)
    logging.info("Slits persisted")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )
    main()
