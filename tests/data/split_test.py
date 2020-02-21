import unittest
from datetime import datetime

from src.data import split


class SplitTest(unittest.TestCase):
    def test_whenCreateSplit_shouldCreate2015AsTestData(self):
        datetimes = [
            datetime(2013, 2, 1),
            datetime(2014, 2, 1),
            datetime(2015, 2, 1),
            datetime(2015, 2, 1),
        ]

        _, _, test_set = split.create_split(datetimes)

        self.assertEqual(len(test_set), 2)

    def test_whenCreateSplit_shouldCreate2014AsValidData(self):
        datetimes = [
            datetime(2013, 2, 1),
            datetime(2014, 2, 1),
            datetime(2014, 2, 1),
            datetime(2015, 2, 1),
        ]

        _, valid_set, _ = split.create_split(datetimes)

        self.assertEqual(len(valid_set), 2)

    def test_whenCreateSplit_eachSetIsUnique(self):
        datetimes = self._create_datetimes()
        train_set, valid_set, test_set = split.create_split(datetimes)

        # Important to counter the shuffle.
        train_set = sorted(train_set)
        valid_set = sorted(valid_set)
        test_set = sorted(test_set)

        self.assertTrue(valid_set not in train_set)
        self.assertTrue(test_set not in train_set)
        self.assertTrue(test_set not in valid_set)

    def test_whenCreateSplit_shouldUseAllValues(self):
        datetimes = self._create_datetimes()

        train_set, valid_set, test_set = split.create_split(datetimes)

        self.assertTrue(len(datetimes), len(train_set) + len(valid_set) + len(test_set))

    def test_canSaveAndLoadSplits(self):
        datetimes = self._create_datetimes()
        train_set, valid_set, test_set = split.create_split(datetimes)

        split.persist_split(train_set, valid_set, test_set, dir_path="/tmp")
        loaded_train_set, loaded_valid_set, loaded_test_set = split.load(
            dir_path="/tmp"
        )

        self.assertEqual(train_set, loaded_train_set)
        self.assertEqual(valid_set, loaded_valid_set)
        self.assertEqual(test_set, loaded_test_set)

    def _create_datetimes(self):
        datetimes = []
        for year in range(2010, 2015):
            for month in range(1, 13):
                for day in range(1, 29):
                    datetimes.append(datetime(year, month, day))

        return datetimes
