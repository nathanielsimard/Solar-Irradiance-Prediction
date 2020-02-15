import time
import unittest
from datetime import datetime

import tensorflow as tf

import src.data.clearskydata as csd
import tests.data.config_test as config_test
from src.data.config import read_configuration_file
from src.data.metadata import Station
from src.data import config
import pandas as pd

DUMMY_TRAIN_CFG_PATH = "tests/data/samples/dummy_train_cfg.json"


class ClearSkyDataTest(unittest.TestCase):
    def test_dataloader_instanciation(self):
        dataset = self._create_data_loader()
        self.assertIsInstance(dataset, tf.data.Dataset)

    def test_sample_shapes(self):
        dataset = self._create_data_loader(target_datetimes=[])
        for (meta, image, _) in dataset:
            self.assertTrue(meta.shape == (10, 10))

    def assertCloseTo(self, value: float, target: float, epsilon: float = 0.001):
        """ Check if a value is close to another, between +- epsilon.

        Arguments:
            value {float} -- value to test
            target {float} -- value wanted

        Keyword Arguments:
            epsilon {float} -- [+- range for the value] (default: {0.001})
        """
        self.assertGreater(value, target - epsilon)
        self.assertLess(value, target + epsilon)

    def test_precompute(self):
        meta_config = config.read_configuration_file(
            "tests/data/samples/train_config_raphael.json"
        )
        target_datetimes = pd.Series(meta_config.target_datetimes)
        stations = meta_config.stations
        clearsky = csd.Clearsky()
        clearsky._precompute_clearsky_values(target_datetimes, stations)
        self.assertCloseTo(
            clearsky.cache["40.0519;-88.3731;230.00;2015-01-04 19:15:00"], 414.271049
        )
        self.assertCloseTo(
            clearsky.cache["40.0519;-88.3731;230.00;2014-07-10 00:15:00"], 98.633557
        )

    def test_clearsky_prediction(self):
        dataset = self._create_data_loader(
            target_datetimes=[datetime(2010, 6, 19, 22, 15)]
        )

        for (meta, image, target) in dataset:
            print(meta[0, 0])
            self.assertCloseTo(meta[0, csd.CSMDOffset.GHI_T].numpy(), 471.675670)
            self.assertCloseTo(meta[0, csd.CSMDOffset.GHI_T_1h].numpy(), 280.165857)
            self.assertCloseTo(meta[0, csd.CSMDOffset.GHI_T_3h].numpy(), 0.397029)
            self.assertCloseTo(meta[0, csd.CSMDOffset.GHI_T_6h].numpy(), 0.0)
            pass

    def test_clearsky_prediction_function(self):
        target_datetime = datetime(2010, 6, 19, 22, 15)
        config = read_configuration_file(config_test.DUMMY_TEST_CFG_PATH)
        preditions = csd.Clearsky().calculate_clearsky_values(
            config.stations[Station.BND], target_datetime
        )
        self.assertCloseTo(preditions[csd.CSMDOffset.GHI_T], 471.675670)
        self.assertCloseTo(preditions[csd.CSMDOffset.GHI_T_1h], 280.165857)
        self.assertCloseTo(preditions[csd.CSMDOffset.GHI_T_3h], 0.397029)
        self.assertCloseTo(preditions[csd.CSMDOffset.GHI_T_6h], 0.0)

    def test_clearsky_cache_key(self):
        cs = csd.Clearsky()
        target_datetime = datetime(2010, 6, 19, 22, 15)
        config = read_configuration_file(config_test.DUMMY_TEST_CFG_PATH)
        key = cs._generate_cache_key(target_datetime, config.stations[Station.BND])
        self.assertEqual(key, "40.0519;-88.3731;230.00;2010-06-19 22:15:00")

    @unittest.skip("Not essential")
    def test_clearsky_prediction_uncached_performance(self):
        target_datetime = datetime(2010, 6, 19, 22, 15)
        config = read_configuration_file(config_test.DUMMY_TEST_CFG_PATH)
        start_time = time.time()
        iterations = 100
        for i in range(0, iterations):
            csd.Clearsky(clear_cache=True).calculate_clearsky_values(
                config.stations[Station.BND], target_datetime
            )
        delta = time.time() - start_time
        iterations_per_seconde = iterations / delta
        self.assertLess(delta, 4)  # 3.59s on i5 8600k
        self.assertGreater(iterations_per_seconde, 25)  # 27.57 IPS. Too slow!

    @unittest.skip("Not essential")
    def test_clearsky_prediction_cached_performance(self):
        target_datetime = datetime(2010, 6, 19, 22, 15)
        config = read_configuration_file(config_test.DUMMY_TEST_CFG_PATH)
        start_time = time.time()
        iterations = 1000
        cs = csd.Clearsky()
        for i in range(0, iterations):
            cs.calculate_clearsky_values(config.stations[Station.BND], target_datetime)
        delta = time.time() - start_time
        iterations_per_seconde = iterations / delta
        self.assertLess(delta, 0.1)  # 0.058s on i5 8600k
        self.assertGreater(iterations_per_seconde, 25)  # 17 196 IPS. Ok!

    def test_clearsky_targets(self):
        dataset = self._create_data_loader(target_datetimes=[datetime(2010, 1, 1, 13)])

        for (meta, image, target) in dataset:
            print(target[0, :])
            self.assertCloseTo(target[0, csd.Targets.GHI_T], -3.58)
            self.assertCloseTo(target[0, csd.Targets.GHI_T_1h], 29.106667)
            self.assertCloseTo(target[0, csd.Targets.GHI_T_3h], 356.273333)
            self.assertCloseTo(target[0, csd.Targets.GHI_T_6h], 481.046667)
            pass

    @unittest.skip("Need rewriting")
    def test_clearsky_dataloader(self):
        dataset = self._create_data_loader()
        # target_datetimes = (
        #     dataset.catalog[50:80].index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        # )
        # dataset = csd.prepare_dataloader(
        #     catalog, target_datetimes, stations, target_time_offsets, None
        # )
        for (meta, image, target) in dataset:
            self.assertTrue(meta.numpy().shape == (30, len(csd.CSMDOffset)))
            self.assertTrue(
                image.numpy().shape == (30, 64, 64, 5)
            )  # TODO: Rendre la taille de l'image paramétrable
            self.assertTrue(target.numpy().shape == (30, 4))
            self.assertTrue(
                image.numpy().shape == (30, 64, 64, 5)
            )  # TODO: Rendre la taille de l'image paramétrable
            self.assertTrue(target.numpy().shape == (30, 4))

    def _create_data_loader(self, target_datetimes=None):
        config = read_configuration_file(DUMMY_TRAIN_CFG_PATH)

        if target_datetimes is None:
            target_datetimes = config.target_datetimes

        station = Station.BND

        return csd.prepare_dataloader(
            config.catalog,
            target_datetimes,
            station,
            config.stations[station],
            config.target_time_offsets,
            {},
        )
