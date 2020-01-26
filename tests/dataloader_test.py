import unittest
import src.dataloader as dl
import tensorflow as tf


class BasicDataLoaderUnitTest(unittest.TestCase):
    def _create_data_loader(self):
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file("tests/data/dummy_train_cfg.json")
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
        ) = dl.read_configuration_file("tests/data/dummy_test_cfg.json")
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
        ) = dl.read_configuration_file("tests/data/dummy_train_cfg.json")
        target_datetimes = []
        dataset = dl.prepare_dataloader(
            catalog, target_datetimes, stations, target_time_offsets, None
        )
        for (meta, image) in dataset:
            self.assertTrue(meta.shape == (10, 10))
        #   pass

    def test_clearsky_prediction(self):
        (
            catalog,
            target_datetimes,
            stations,
            target_time_offsets,
        ) = dl.read_configuration_file("tests/data/dummy_train_cfg.json")
        target_datetimes = ["2010-06-19 22:15:00"]
        dataset = dl.prepare_dataloader(
            catalog, target_datetimes, stations, target_time_offsets, None
        )

        for (meta, image, target) in dataset:
            print(meta[0, 0])
            self.assertGreater(
                meta[0, dl.ClearSkyMetaDataOffsset.GHI_T].numpy(), 471.67
            )  # T=0, 471.675670
            self.assertLess(
                meta[0, dl.ClearSkyMetaDataOffsset.GHI_T].numpy(), 471.68
            )  # T=0, 471.675670
            self.assertGreater(
                meta[0, dl.ClearSkyMetaDataOffsset.GHI_T_1h].numpy(), 280.16
            )  # T=1, 280.165857
            self.assertLess(
                meta[0, dl.ClearSkyMetaDataOffsset.GHI_T_1h].numpy(), 280.17
            )
            self.assertGreater(
                meta[0, dl.ClearSkyMetaDataOffsset.GHI_T_3h].numpy(), 96.87
            )  # T=3, 96.875384
            self.assertLess(meta[0, dl.ClearSkyMetaDataOffsset.GHI_T_3h].numpy(), 96.88)
            self.assertGreater(
                meta[0, dl.ClearSkyMetaDataOffsset.GHI_T_6h].numpy(), 0.397
            )  # T=6 0.397029
            self.assertLess(
                meta[0, dl.ClearSkyMetaDataOffsset.GHI_T_6h].numpy(), 0.398
            )  # T=0
            pass
