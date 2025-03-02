#  type: ignore
import argparse
import datetime
import json
import os
import typing

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from src import logging
from src.data import dataloader, preprocessing
from src.data.metadata import Coordinates, MetadataLoader, Station
from src.model import base, conv3d_tran

logger = logging.create_logger(__name__)

BEST_WEIGHTS = "4"


def prepare_dataloader(
    dataframe: pd.DataFrame,
    target_datetimes: typing.List[datetime.datetime],
    station: str,
    coordinates: typing.Tuple[float, float, float],
    target_time_offsets: typing.List[datetime.timedelta],
    config: dataloader.DataloaderConfig,
) -> tf.data.Dataset:
    """Output data.

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
        station: station name of interest
        coordinates: station's coordinates (latitude, longitude, elevation).
            During evaluation time, it will only be one station to avoid confusions.
            See comment on function `generate_all_predictions` with the for loop.
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration for the dataloader.

    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.

    """

    logger.info(f"Prepare dataloader for station {station} and config {config}")
    metadata_loader = MetadataLoader(dataframe=dataframe, training=False)
    metadata_generator = metadata_loader.load(
        Station(station),
        Coordinates(coordinates[0], coordinates[1], coordinates[2]),
        target_datetimes=target_datetimes,
        skip_missing=False,
        num_images=config.num_images,
        time_interval_min=config.time_interval_min,
    )

    return dataloader.create_dataset(
        lambda: metadata_generator, config=config, enable_image_cache=False,
    )


def prepare_model(
    target_time_offsets: typing.List[datetime.timedelta],
    config: typing.Dict[typing.AnyStr, typing.Any],
) -> base.Model:
    """Model for the data.

    See https://github.com/mila-iqia/ift6759/tree/master/projects/project1/evaluation.md for more information.
    Args:
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A ``base.Model`` object that can be used to generate new GHI predictions given imagery tensors.

    """
    model = conv3d_tran.CNN3DTranClearsky()
    model.load(BEST_WEIGHTS)
    logger.info(f"Loaded model: {model.title}")
    return model


def generate_predictions(
    data_loader: tf.data.Dataset, model: tf.keras.Model, pred_count: int
) -> np.ndarray:
    """Generate and returns model predictions given the data prepared by a data loader."""
    predictions = []
    scaling_ghi = preprocessing.min_max_scaling_ghi()
    with tqdm.tqdm("generating predictions", total=pred_count) as pbar:
        for iter_idx, minibatch in enumerate(data_loader.batch(64)):
            logger.info(f"Minibatch #{iter_idx}")
            assert (
                isinstance(minibatch, tuple) and len(minibatch) >= 2
            ), "the data loader should load each minibatch as a tuple with model input(s) and target tensors"
            # Call the model without the target and with training = False.
            pred = model(minibatch[0:-1]).numpy()

            # Rescale the GHI values.
            pred = scaling_ghi.original(pred)
            assert (
                pred.ndim == 2
            ), "prediction tensor shape should be BATCH x SEQ_LENGTH"
            predictions.append(pred)
            pbar.update(len(pred))
    return np.concatenate(predictions, axis=0)


def generate_all_predictions(
    target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    target_datetimes: typing.List[datetime.datetime],
    target_time_offsets: typing.List[datetime.timedelta],
    dataframe: pd.DataFrame,
    user_config: typing.Dict[typing.AnyStr, typing.Any],
) -> np.ndarray:
    """Generate and returns model predictions given the data prepared by a data loader."""
    # we will create one data loader per station to make sure we avoid mixups in predictions
    logger.info("Generating all prefictions")
    predictions = []
    for station_idx, station_name in enumerate(target_stations):
        # usually, we would create a single data loader for all stations, but we just want to avoid trouble...
        print(
            f"preparing data loader & model for station '{station_name}' ({station_idx + 1}/{len(target_stations)})"
        )
        coordinates = target_stations[station_name]

        # Create the model
        model = prepare_model(target_time_offsets, user_config)
        # Get the configuration from the model to load the proper dataset.
        model_config = model.config()
        # When an error occured during loading the data (Like missing target).
        # We should not crash nor skip the datetimel.
        model_config.error_strategy = dataloader.ErrorStrategy.ignore
        model_config.filter_night = False

        dataset = prepare_dataloader(
            dataframe,
            target_datetimes,
            station_name,
            coordinates,
            target_time_offsets,
            model_config,
        )
        # Apply the preprocessing needed for the model.
        dataset = model.preprocess(dataset)

        station_preds = generate_predictions(
            dataset, model, pred_count=len(target_datetimes)
        )
        assert len(station_preds) == len(
            target_datetimes
        ), "number of predictions mismatch with requested datetimes"
        predictions.append(station_preds)
    return np.concatenate(predictions, axis=0)


def parse_gt_ghi_values(
    target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    target_datetimes: typing.List[datetime.datetime],
    target_time_offsets: typing.List[datetime.timedelta],
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parse all required station GHI values from the provided dataframe for the evaluation of predictions."""
    gt = []
    for station_idx, station_name in enumerate(target_stations):
        station_ghis = dataframe[station_name + "_GHI"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_ghis.index:
                    seq_vals.append(
                        station_ghis.iloc[station_ghis.index.get_loc(index)]
                    )
                else:
                    seq_vals.append(float("nan"))
            gt.append(seq_vals)
    return np.concatenate(gt, axis=0)


def parse_nighttime_flags(
    target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    target_datetimes: typing.List[datetime.datetime],
    target_time_offsets: typing.List[datetime.timedelta],
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parse all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        station_flags = dataframe[station_name + "_DAYTIME"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_flags.index:
                    seq_vals.append(
                        station_flags.iloc[station_flags.index.get_loc(index)] > 0
                    )
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def main(
    preds_output_path: typing.AnyStr,
    admin_config_path: typing.AnyStr,
    user_config_path: typing.Optional[typing.AnyStr] = None,
    stats_output_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    """Extract predictions from a user model/data loader combo and saves them to a CSV file."""
    user_config = {}
    if user_config_path:
        assert os.path.isfile(
            user_config_path
        ), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(
        admin_config_path
    ), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    if "start_bound" in admin_config:
        dataframe = dataframe[
            dataframe.index
            >= datetime.datetime.fromisoformat(admin_config["start_bound"])
        ]
    if "end_bound" in admin_config:
        dataframe = dataframe[
            dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])
        ]

    target_datetimes = [
        datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]
    ]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [
        pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]
    ]

    if (
        "bypass_predictions_path" in admin_config
        and admin_config["bypass_predictions_path"]
    ):
        # re-open cached output if possible (for 2nd pass eval)
        assert os.path.isfile(
            preds_output_path
        ), f"invalid preds file path: {preds_output_path}"
        with open(preds_output_path, "r") as fd:
            predictions = fd.readlines()
        assert len(predictions) == len(target_datetimes) * len(
            target_stations
        ), "predicted ghi sequence count mistmatch wrt target datetimes x station count"
        assert len(predictions) % len(target_stations) == 0
        predictions = np.asarray(
            [float(ghi) for p in predictions for ghi in p.split(",")]
        )
    else:
        predictions = generate_all_predictions(
            target_stations,
            target_datetimes,
            target_time_offsets,
            dataframe,
            user_config,
        )
        with open(preds_output_path, "w") as fd:
            for pred in predictions:
                fd.write(",".join([f"{v:0.03f}" for v in pred.tolist()]) + "\n")

    if any([s + "_GHI" not in dataframe for s in target_stations]):
        print("station GHI measures missing from dataframe, skipping stats output")
        return

    assert not np.isnan(
        predictions
    ).any(), "user predictions should NOT contain NaN values"
    predictions = predictions.reshape(
        (len(target_stations), len(target_datetimes), len(target_time_offsets))
    )
    gt = parse_gt_ghi_values(
        target_stations, target_datetimes, target_time_offsets, dataframe
    )
    gt = gt.reshape(
        (len(target_stations), len(target_datetimes), len(target_time_offsets))
    )
    day = parse_nighttime_flags(
        target_stations, target_datetimes, target_time_offsets, dataframe
    )
    day = day.reshape(
        (len(target_stations), len(target_datetimes), len(target_time_offsets))
    )

    squared_errors = np.square(predictions - gt)
    stations_rmse = np.sqrt(np.nanmean(squared_errors, axis=(1, 2)))
    for station_idx, (station_name, station_rmse) in enumerate(
        zip(target_stations, stations_rmse)
    ):
        print(f"station '{station_name}' RMSE = {station_rmse:.02f}")
    horizons_rmse = np.sqrt(np.nanmean(squared_errors, axis=(0, 1)))
    for horizon_idx, (horizon_offset, horizon_rmse) in enumerate(
        zip(target_time_offsets, horizons_rmse)
    ):
        print(f"horizon +{horizon_offset} RMSE = {horizon_rmse:.02f}")
    overall_rmse = np.sqrt(np.nanmean(squared_errors))
    print(f"overall RMSE = {overall_rmse:.02f}")

    if stats_output_path is not None:
        # we remove nans to avoid issues in the stats comparison script, and focus on daytime predictions
        squared_errors = squared_errors[~np.isnan(gt) & day]
        with open(stats_output_path, "w") as fd:
            for err in squared_errors.reshape(-1):
                fd.write(f"{err:0.03f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "preds_out_path",
        type=str,
        help="path where the raw model predictions should be saved (for visualization purposes)",
    )
    parser.add_argument(
        "admin_cfg_path",
        type=str,
        help="path to the JSON config file used to store test set/evaluation parameters",
    )
    parser.add_argument(
        "-u",
        "--user_cfg_path",
        type=str,
        default=None,
        help="path to the JSON config file used to store user model/dataloader parameters",
    )
    parser.add_argument(
        "-s",
        "--stats_output_path",
        type=str,
        default=None,
        help="path where the prediction stats should be saved (for benchmarking)",
    )
    args = parser.parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
        stats_output_path=args.stats_output_path,
    )
