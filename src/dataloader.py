import datetime
import json
import os
import typing
import data
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3

from src import data

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


import pickle
import json

from enum import IntEnum

class ClearSkyMetaDataOffsset(IntEnum):
    GHI_T = 0
    GHI_T_1h = 1
    GHI_T_3h = 2
    GHI_T_6h = 3

def read_configuration_file(filename):
    """ Reads the configuration file as specified in the evaluation guidelines.

    Returns the parameters required for the prepare_dataloader method, which are:
    - The dataframe
    - The list of target date times (UTC)
    - A dictionnary of stations and their lat, long and altitude.
    - The target time offsets
    
    """
    with open(filename) as json_file:
        configuration = json.load(json_file)
    
    catalog_path = configuration['dataframe_path']
    catalog = pickle.load(open(catalog_path,"rb"))
    stations = configuration['stations']
    target_datetimes = configuration['target_datetimes']
    target_time_offsets = configuration['target_time_offsets']
    return (catalog, target_datetimes, stations, target_time_offsets)


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    """This function should be modified in order to prepare & return your own data loader.
    Note that you can use either the netCDF or HDF5 data. Each iteration over your data loader should return a
    2-element tuple containing the tensor that should be provided to the model as input, and the target values. In
    this specific case, you will not be able to provide the latter since the dataframe contains no GHI, and we are
    only interested in predictions, not training. Therefore, you must return a placeholder (or ``None``) as the second
    tuple element.
    Reminder: the dataframe contains imagery paths for every possible timestamp requested in ``target_datetimes``.
    However, we expect that you will use some of the "past" imagery (i.e. imagery at T<=0) for any T in
    ``target_datetimes``, but you should NEVER rely on "future" imagery to generate predictions (for T>0). We
    will be inspecting data loader implementations to ensure this is the case, and those who "cheat" will be
    dramatically penalized.
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
    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """
    ################################## MODIFY BELOW ##################################
    # As a starting point, the basic data generator will provide:
    # 
    # x values:
    # 

    def clearsky_data_generator():
        """
        This will generate data for a baseline clearsky model.

        Picture data will not be read in the first iteration. 
        """

        meta_loader = data.MetadataLoader(file_name = None, dataframe = dataframe)

        batch_size = 32
        image_dim = (64, 64)
        n_channels = 5
        output_seq_len = 4

        for i in range(0, len(target_datetimes), batch_size):
            batch_of_datetimes = target_datetimes[i:i+batch_size]
            meta_data_loader = meta_loader.load(data.Station.BND,target_datetimes=batch_of_datetimes)
            meta_data = np.zeros((len(batch_of_datetimes),10))
            samples = tf.random.uniform(shape=(
                len(batch_of_datetimes), image_dim[0], image_dim[1], n_channels
            ))
            j=0
            print(len(batch_of_datetimes))
            for sample in meta_data_loader:
                bnd = Location(latitude = sample.latitude, longitude = sample.longitude, altitude = sample.altitude)
                future_clearsky_ghi = bnd.get_clearsky( pd.date_range(start=batch_of_datetimes[j], periods=7, freq="1H"))["ghi"]
                print(bnd, future_clearsky_ghi)
                meta_data[j,0] = future_clearsky_ghi[ClearSkyMetaDataOffsset.GHI_T] #T=0
                meta_data[j,1] = future_clearsky_ghi[ClearSkyMetaDataOffsset.GHI_T_1h] #T=T+1
                meta_data[j,2] = future_clearsky_ghi[ClearSkyMetaDataOffsset.GHI_T_3h] #T=T+3
                meta_data[j,3] = future_clearsky_ghi[ClearSkyMetaDataOffsset.GHI_T_6h] #T=T+7
                j=j+1
            targets = tf.zeros(shape=(
                len(batch_of_datetimes), output_seq_len
            ))
            # Remember that you do not have access to the targets.
            # Your dataloader should handle this accordingly.
            #yield (tf.convert_to_tensor(meta_data), samples), targets
            yield tf.convert_to_tensor(meta_data), samples, targets

    data_loader = tf.data.Dataset.from_generator(
        #clearsky_data_generator, ((tf.float32, tf.float32), tf.float32)
        clearsky_data_generator, (tf.float32, tf.float32, tf.float32)
    )

    ################################### MODIFY ABOVE ##################################

    return data_loader

#import pickle
#catalog = pickle.load(open("tests/data/catalog-test.pkl","rb"))
#g = prepare_dataloader(catalog)
