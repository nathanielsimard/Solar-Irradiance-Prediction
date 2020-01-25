import unittest

from src.data import MetadataLoader


class MetadataLoaderTest(unittest.TestCase):
    def test_load_stations_load_image(self):
        """Should load the right first_image considering the compression."""
        for (test_name, compression, expected) in [
            (
                "8 bits Compression",
                "8bit",
                "/project/cq-training-1/project1/data/hdf5v7_8bit/2010.01.01.0800.h5",
            ),
            (
                "16 bits Compression",
                "16bit",
                "/project/cq-training-1/project1/data/hdf5v5_16bit/2010.01.01.0800.h5",
            ),
        ]:
            with self.subTest(test_name):
                loader = MetadataLoader("tests/data/catalog-test.pkl")

                metadata = loader.load_stations(compression=compression)

                first_image = metadata[0].image
                print(type(first_image))
                self.assertEqual(expected, first_image)

    def test_load_stations_night_time(self):
        """Should only load the metadata considering the night time."""
        for (test_name, night_time, num_metadata) in [
            ("Load all metadata", False, 2068),
            ("Don't Load night metadata", True, 1080),
        ]:
            with self.subTest(test_name):
                loader = MetadataLoader("tests/data/catalog-test.pkl")

                metadata = loader.load_stations(night_time=night_time)

                if night_time:
                    for m in metadata:
                        self.assertTrue(m.night_time)

                self.assertEqual(num_metadata, len(metadata))
