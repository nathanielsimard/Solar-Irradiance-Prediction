import os
import unittest

from src import env


class EnvTest(unittest.TestCase):
    """ Local test are not covered since they don't have any impact on the team.
    """

    def test_tensorboard_defaut_path(self):
        env.run_local = False
        expected_base_dir = "/project/cq-training-1/project1/teams/team10/tensorboard/"
        self.assertEqual(
            env.get_tensorboard_log_directory()[0 : len(expected_base_dir)],
            expected_base_dir,
        )

    def test_default_split_path(self):
        env.run_local = False
        self.assertEqual(
            env.get_split_path(), "/project/cq-training-1/project1/teams/team10/split"
        )

    def test_get_catalog_path(self):
        env.run_local = False
        self.assertEqual(
            env.get_catalog_path(),
            "/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl",
        )

    def test_default_tf_cache_file(self):
        env.run_local = False
        self.assertEqual(
            env.get_tf_cache_file(),
            "/project/cq-training-1/project1/teams/team10/cached/cached",
        )

    def test_default_image_reader_directory(self):
        env.run_local = False
        self.assertEqual(env.get_image_reader_cache_directory(), "/tmp")

    def test_local_image_reader_directory(self):
        env.run_local = True
        self.assertEqual(
            env.get_image_reader_cache_directory(), "../image_reader_cache"
        )

    def test_slurm_image_reader_directory(self):
        env.run_local = False
        os.environ["SLURM_TMPDIR"] = "/test"
        self.assertEqual(
            env.get_image_reader_cache_directory(), "/test/image_reader_cache"
        )
        del os.environ["SLURM_TMPDIR"]
