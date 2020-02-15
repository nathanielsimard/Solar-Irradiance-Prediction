from src import logging
import tensorflow as tf
from src.data.train import load_data_and_create_generators
import time as time
logger = logging.create_logger(__name__)


def run(
    enable_tf_caching=False, skip_non_cached=False,
):
    """Performs a dry run with the data generators."""
    logger.info("Dry Run.")
    # Only test the generators, for debugging weird behavior and corner cases.
    (
        train_generator,
        valid_generator,
        test_generator,
    ) = load_data_and_create_generators(
        enable_tf_caching=enable_tf_caching, skip_non_cached=skip_non_cached
    )
    samples = 0
    start = time.time()
    for sample in train_generator:
        samples += 1
        if(samples % 100 == 0):
            temps = time.time() - start
            print(f"{100/temps} images par secondes {tf.shape(sample[0])}")
            start = time.time()
