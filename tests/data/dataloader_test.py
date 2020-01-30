import unittest
from datetime import datetime

import tensorflow as tf

import src.data.dataloader as dl
from src.data.config import read_configuration_file

DUMMY_TRAIN_CFG_PATH = "tests/data/dummy_train_cfg.json"


class BasicDataLoaderTest(unittest.TestCase):
    pass


class BasicDataLoaderUnitTest(unittest.TestCase):
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

    def test_clearsky_prediction(self):
        dataset = self._create_data_loader(
            target_datetimes=[datetime(2010, 6, 19, 22, 15)]
        )

        for (meta, image, target) in dataset:
            print(meta[0, 0])
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T].numpy(), 471.675670)
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T_1h].numpy(), 280.165857)
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T_3h].numpy(), 0.397029)
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T_6h].numpy(), 0.0)
            pass

    def test_clearsky_targets(self):
        dataset = self._create_data_loader(target_datetimes=[datetime(2010, 1, 1, 13)])

        for (meta, image, target) in dataset:
            print(target[0, :])
            self.assertCloseTo(target[0, dl.Targets.GHI_T], -3.58)
            self.assertCloseTo(target[0, dl.Targets.GHI_T_1h], 29.106667)
            self.assertCloseTo(target[0, dl.Targets.GHI_T_3h], 356.273333)
            self.assertCloseTo(target[0, dl.Targets.GHI_T_6h], 481.046667)
            pass

    #    def test_clearsky_dataloader(self):
    #        dataset = self._create_data_loader()
    #
    #        target_datetimes = catalog[50:80].index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    #        dataset = dl.prepare_dataloader(
    #            catalog, target_datetimes, stations, target_time_offsets, None
    #        )
    #        for (meta, image, target) in dataset:
    #            self.assertTrue(meta.numpy().shape == (30, len(dl.CSMDOffset)))
    #            self.assertTrue(
    #                image.numpy().shape == (30, 64, 64, 5)
    #            )  # TODO: Rendre la taille de l'image paramétrable
    #            self.assertTrue(target.numpy().shape == (30, 4))
    #            self.assertTrue(
    #                image.numpy().shape == (30, 64, 64, 5)
    #            )  # TODO: Rendre la taille de l'image paramétrable
    #            self.assertTrue(target.numpy().shape == (30, 4))

    def _create_data_loader(self, target_datetimes=None):
        config = read_configuration_file(DUMMY_TRAIN_CFG_PATH)

        if target_datetimes is None:
            target_datetimes = config.target_datetimes

        return dl.prepare_dataloader(
            config.catalog,
            target_datetimes,
            config.stations,
            config.target_time_offsets,
            {},
        )
