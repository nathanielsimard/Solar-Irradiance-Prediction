import datetime
import typing
from enum import IntEnum
from typing import Generator

import numpy as np
import pandas as pd
import tensorflow as tf
from pvlib.location import Location

from src.data import metadata
from src.data.config import Config


class DataLoader(object):
    def create_dataset(
        self, metadata_generator=Generator[metadata.Metadata, None, None]
    ) -> tf.data.Dataset:
        pass


def create_dataloader(config: Config) -> DataLoader:
    # metadata_loader = metadata.MetadataLoader(dataframe=config.catalog)
    return DataLoader()


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


def prepare_dataloader(
    dataframe: pd.DataFrame,
    target_datetimes: typing.List[datetime.datetime],
    stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    target_time_offsets: typing.List[datetime.timedelta],
    config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    """Output an augmented dataset for the GHI prediction.

    See https://github.com/mila-iqia/ift6759/tree/master/projects/project1/evaluation.md for more information.
    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
            relevant timestamp values over the test period.
        target_datetimes: a list of timestamps that your data loader should use to provide imagery for your model.
            The ordering of this list is important, as each element corresponds to a sequence of GHI values
            to predict. By definition, the GHI values must be provided for the offsets given by ``target_time_offsets``
            which are added to each timestamp (T=0) in this datetimes list.
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns
    -------
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.

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
        # TODO: Add support for other stations than BND
        for i in range(0, len(target_datetimes), batch_size):
            batch_of_datetimes = target_datetimes[i : i + batch_size]
            meta_data_loader = meta_loader.load(
                metadata.Station.BND, target_datetimes=batch_of_datetimes
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
                    latitude=sample.latitude,
                    longitude=sample.longitude,
                    altitude=sample.altitude,
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
