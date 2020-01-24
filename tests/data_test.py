import unittest

from src.data import MetadataLoader


class MetadataLoaderTest(unittest.TestCase):
    """Dummy Test Class."""

    def test_load_all_stations(self):
        for (test_name, compression, expected) in [
            (
                "8 bits Compression",
                "8bits",
                "/project/cq-training-1/project1/data/hdf5v7_8bit/2010.01.01.0800.h5",
            ),
            (
                "16 bits Compression",
                "16bits",
                "/project/cq-training-1/project1/data/hdf5v5_16bit/2010.01.01.0800.h5",
            ),
        ]:
            with self.subTest(test_name):
                loader = MetadataLoader("tests/data/catalog-test.pkl")

                metadata = loader.load_all_stations(compression=compression)

                image = metadata[0].image
                self.assertEqual(expected, image)
