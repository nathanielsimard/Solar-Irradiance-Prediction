from enum import IntEnum
import tensorflow as tf
import typing
import datetime
import numpy as np
import pandas as pd
import pickle
from pvlib.location import Location


from src.data import metadata

from typing import Dict
from src.data.metadata import Coordinates, Station

from src import logging


logger = logging.create_logger(__name__)


class CSMDOffset(IntEnum):
    """Mapping for the metadata to the location in the tensor."""

    # TODO: Find python equivalent of "c" enums.
    GHI_T = 0
    GHI_T_1h = 1
    GHI_T_3h = 2
    GHI_T_6h = 3


class Targets(IntEnum):
    """Mapping for the targets to their location in the target tensor."""

    GHI_T = 0
    GHI_T_1h = 1
    GHI_T_3h = 2
    GHI_T_6h = 3


class Clearsky:
    """Clearsky Prediction Handling Class.

    Now with crude caching! (Will take about 1gb on disk / ram)

    """

    def __init__(self, clear_cache=False, enable_caching=False):
        """Constructor for the cached clearsky.

        Keyword Arguments:
            clear_cache {bool} -- [Force to clear the cache] (default: {False})
        """
        self.cache = {}  # Crude caching
        self.enable_caching = enable_caching
        self.cache_filename = "clearcky_cache.pkl"
        if enable_caching:
            try:
                with open(self.cache_filename, "rb") as file:
                    logger.info(
                        f"Loading precomputed clearsky values from {self.cache_filename}"
                    )
                    self.cache = pickle.load(file)
            except (FileNotFoundError, EOFError):
                logger.info("No clearsky cache found or corrupted cache!")
            if clear_cache:
                self.cache.clear()

    def __del__(self):
        """Delete the objects and close the cache."""

    def clear_cache(self):
        """Forces clearing the cache."""
        if self.enable_caching:
            self.cache.clear()

    def _load_values_to_cache(self, row):
        clearsky = np.zeros(4)
        clearsky[0] = row["ghi"]
        clearsky[1] = row["ghi_t1"]
        clearsky[2] = row["ghi_t3"]
        clearsky[3] = row["ghi_t6"]
        self.cache[row["cache_key"]] = clearsky

    def _precompute_clearsky_values(
        self, target_datetimes=None, stations: Dict[Station, Coordinates] = None
    ):
        if len(self.cache) > 1000000:
            logger.info(
                """1000000 values already cached. Assuming everything is cached, skipping pre-computing.
                   Flush the cache to force precomputation"""
            )
            return
        if target_datetimes is not None:
            if stations is None:
                raise ValueError(
                    "Must provided stations along with target datetimes for pre-computation"
                )

            for station in stations:
                logger.info(
                    f"Precomputing clearsky values for {station} located at {stations[station]}"
                )
                coordinates = stations[station]
                location = Location(
                    latitude=coordinates.latitude,
                    longitude=coordinates.longitude,
                    altitude=coordinates.altitude,
                )
                # Computing values for 4 increments of time. (0, +1h, +3h, +6h)
                target_datetimes = pd.Series(target_datetimes)
                clearsky_values_t0 = location.get_clearsky(
                    pd.DatetimeIndex(target_datetimes)
                )
                clearsky_values_t1 = location.get_clearsky(
                    pd.DatetimeIndex(target_datetimes + pd.Timedelta(1, unit="h"))
                )
                clearsky_values_t3 = location.get_clearsky(
                    pd.DatetimeIndex(target_datetimes + pd.Timedelta(3, unit="h"))
                )
                clearsky_values_t6 = location.get_clearsky(
                    pd.DatetimeIndex(target_datetimes + pd.Timedelta(6, unit="h"))
                )

                clearsky_values_t0[
                    "cache_key"
                ] = clearsky_values_t0.index.to_series().apply(
                    self._generate_cache_key, args=[coordinates]
                )
                clearsky_values = clearsky_values_t0.reset_index()
                clearsky_values["ghi_t1"] = clearsky_values_t1.reset_index()["ghi"]
                clearsky_values["ghi_t3"] = clearsky_values_t3.reset_index()["ghi"]
                clearsky_values["ghi_t6"] = clearsky_values_t6.reset_index()["ghi"]
                clearsky_values.apply(self._load_values_to_cache, axis=1)
            with open(self.cache_filename, "wb") as file:
                logger.info(
                    f"Saving precomputed clearsky values to {self.cache_filename}"
                )
                pickle.dump(self.cache, file)

    def _generate_cache_key(self, timestamp, coordinates):
        ts_str = str(timestamp)
        return f"{coordinates.latitude:.4f};{coordinates.longitude:.4f};{coordinates.altitude:.2f};{ts_str}"

    def calculate_clearsky_values(
        self, coordinates: metadata.Coordinates, timestamp: pd.Timestamp
    ) -> np.array:
        """Get a numpy array for clearsky values.

        Args:
            coordinates {metadata.Coordinates} : Coordinates of the station.
            timestamp {pd.Timestamp} : Time at which the model should be evaluated

        Returns:
            np.array:-- A numpy array with the computed values at T, T+1, T+3 and T+6 hours.
        """
        cache_key = self._generate_cache_key(timestamp, coordinates)
        if cache_key not in self.cache:
            location = Location(
                latitude=coordinates.latitude,
                longitude=coordinates.longitude,
                altitude=coordinates.altitude,
            )
            future_clearsky_ghi = location.get_clearsky(
                pd.date_range(start=timestamp, periods=7, freq="1H")
            )["ghi"].to_numpy()
            clearsky = np.zeros(len(CSMDOffset))
            clearsky[CSMDOffset.GHI_T] = future_clearsky_ghi[0]  # T=0
            clearsky[CSMDOffset.GHI_T_1h] = future_clearsky_ghi[1]  # T=T+1
            clearsky[CSMDOffset.GHI_T_3h] = future_clearsky_ghi[3]  # T=T+3
            clearsky[CSMDOffset.GHI_T_6h] = future_clearsky_ghi[6]  # T=T+6
            if self.enable_caching:
                self.cache[cache_key] = clearsky
            return clearsky
        else:
            return self.cache[cache_key]


def prepare_dataloader(
    dataframe: pd.DataFrame,
    target_datetimes: typing.List[datetime.datetime],
    station: str,
    coordinates: metadata.Coordinates,
    target_time_offsets: typing.List[datetime.timedelta],
    config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    """Dataloader function of the TAs. See evaluator.py for more.

    Used early in development for testing.

    """

    def clearsky_data_generator():
        """Generate data for a baseline clearsky model.

        Picture data will not be read in the initial branch.
        """
        meta_loader = metadata.MetadataLoader(dataframe=dataframe)

        batch_size = 32
        image_dim = (64, 64)
        n_channels = 5
        output_seq_len = len(Targets)
        for i in range(0, len(target_datetimes), batch_size):
            batch_of_datetimes = target_datetimes[i : i + batch_size]
            meta_data_loader = meta_loader.load(
                station, coordinates, target_datetimes=batch_of_datetimes,
            )
            meta_data = np.zeros((len(batch_of_datetimes), len(CSMDOffset)))
            targets = np.zeros((len(batch_of_datetimes), output_seq_len))
            # TODO : Read the hd5 file and center crop it here
            samples = tf.random.uniform(
                shape=(len(batch_of_datetimes), image_dim[0], image_dim[1], n_channels)
            )
            j = 0
            for sample in meta_data_loader:
                bnd = Location(
                    latitude=sample.coordinates.latitude,
                    longitude=sample.coordinates.longitude,
                    altitude=sample.coordinates.altitude,
                )
                future_clearsky_ghi = bnd.get_clearsky(
                    pd.date_range(start=batch_of_datetimes[j], periods=7, freq="1H")
                )["ghi"]
                # Handle metadata and feature augementation
                meta_data[j, CSMDOffset.GHI_T] = future_clearsky_ghi[0]  # T=0
                meta_data[j, CSMDOffset.GHI_T_1h] = future_clearsky_ghi[1]  # T=T+1
                meta_data[j, CSMDOffset.GHI_T_3h] = future_clearsky_ghi[3]  # T=T+3
                meta_data[j, CSMDOffset.GHI_T_6h] = future_clearsky_ghi[6]  # T=T+7
                # Handle target values
                targets[j, Targets.GHI_T] = sample.target_ghi
                targets[j, Targets.GHI_T_1h] = sample.target_ghi_1h
                targets[j, Targets.GHI_T_3h] = sample.target_ghi_3h
                targets[j, Targets.GHI_T_6h] = sample.target_ghi_6h
                j = j + 1
            # Remember that you do not have access to the targets.
            # Your dataloader should handle this accordingly.
            # yield (tf.convert_to_tensor(meta_data), samples), targets
            yield tf.convert_to_tensor(meta_data), samples, tf.convert_to_tensor(
                targets
            )

    data_loader = tf.data.Dataset.from_generator(
        clearsky_data_generator, (tf.float32, tf.float32, tf.float32)
    )

    return data_loader
