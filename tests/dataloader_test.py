import unittest
import src.dataloader as dl
import tensorflow as tf

DUMMY_TRAIN_CFG_PATH = "tests/data/dummy_train_cfg.json"
DUMMY_TEST_CFG_PATH = "tests/data/dummy_test_cfg.json"


class BasicDataLoaderUnitTest(unittest.TestCase):
    def _create_data_loader(self):
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file(DUMMY_TRAIN_CFG_PATH)
        dataset = dl.prepare_dataloader(
            catalog, target_datetimes, stations, target_time_offsets, None
        )
        return dataset

    def test_read_configuration_file(self):
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file(DUMMY_TEST_CFG_PATH)
        # A few spot checks
        self.assertEquals(len(catalog), 35040)
        self.assertEquals(target_datetimes[10], "2015-01-06T16:00:00")
        self.assertAlmostEquals(stations["BND"][1], -88.37309)
        self.assertEquals(
            target_time_offsets,
            ["P0DT0H0M0S", "P0DT1H0M0S", "P0DT3H0M0S", "P0DT6H0M0S"],
        )

    def test_dataloader_instanciation(self):
        dataset = self._create_data_loader()
        self.assertIsInstance(dataset, tf.data.Dataset)

    def test_read_first_sample(self):
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file(DUMMY_TRAIN_CFG_PATH)
        target_datetimes = []
        dataset = dl.prepare_dataloader(
            catalog, target_datetimes, stations, target_time_offsets, None
        )
        for (meta, image) in dataset:
            self.assertTrue(meta.shape == (10, 10))
        #   pass

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
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file(DUMMY_TRAIN_CFG_PATH)
        target_datetimes = ["2010-06-19 22:15:00"]
        dataset = dl.prepare_dataloader(
            catalog, target_datetimes, stations, target_time_offsets, None
        )

        for (meta, image, target) in dataset:
            print(meta[0, 0])
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T].numpy(), 471.675670)
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T_1h].numpy(), 280.165857)
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T_3h].numpy(), 0.397029)
            self.assertCloseTo(meta[0, dl.CSMDOffset.GHI_T_6h].numpy(), 0.0)
            pass

    def test_clearsky_targets(self):
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file(DUMMY_TRAIN_CFG_PATH)
        target_datetimes = ["2010-01-01 13:00:00"]
        dataset = dl.prepare_dataloader(
            catalog, target_datetimes, stations, target_time_offsets, None
        )

        for (meta, image, target) in dataset:
            print(target[0, :])
            self.assertCloseTo(target[0, dl.Targets.GHI_T], -3.58)
            self.assertCloseTo(target[0, dl.Targets.GHI_T_1h], 29.106667)
            self.assertCloseTo(target[0, dl.Targets.GHI_T_3h], 356.273333)
            self.assertCloseTo(target[0, dl.Targets.GHI_T_6h], 481.046667)
            pass

    def test_clearsky_dataloader(self):
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file(DUMMY_TRAIN_CFG_PATH)

        target_datetimes = catalog[50:80].index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        dataset = dl.prepare_dataloader(
            catalog, target_datetimes, stations, target_time_offsets, None
        )
        for (meta, image, target) in dataset:
            self.assertTrue(meta.numpy().shape == (30, len(dl.CSMDOffset)))
            self.assertTrue(
                image.numpy().shape == (30, 64, 64, 5)
            )  # TODO: Rendre la taille de l'image paramétrable
            self.assertTrue(target.numpy().shape == (30, 4))
