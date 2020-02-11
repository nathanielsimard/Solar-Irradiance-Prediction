import unittest
from src import env


class EnvTest(unittest.TestCase):
    """ Local test are not covered since they don't have any impact on the team.
    """

    def test_tensorboard_defaut_path(self):
        expected_base_dir = "/project/cq-training-1/project1/teams/team10/tensorboard/"
        self.assertEquals(
            env.get_tensorboard_log_directory()[0 : len(expected_base_dir)],
            expected_base_dir,
        )

    def test_default_split_path(self):
        self.assertEquals(
            env.get_split_path(), "/project/cq-training-1/project1/teams/team10/split"
        )

    def test_get_catalog_path(self):
        self.assertEquals(
            env.get_catalog_path(),
            "/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
        )

    def test_default_tf_cache_file(self):
        self.assertEquals(
            env.get_tf_cache_file(),
            "/project/cq-training-1/project1/teams/team10/cached/cached",
        )
