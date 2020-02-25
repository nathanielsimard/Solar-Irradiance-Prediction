from src import logging
from src.data.train import load_data_and_create_generators

logger = logging.create_logger(__name__)


def run(
    enable_tf_caching=False, skip_non_cached=False, model=None
):
    """Performs a dry run with the data generators."""
    logger.info("Dry Run.")
    # Only test the generators, for debugging weird behavior and corner cases.
    (
        train_generator,
        valid_generator,
        test_generator,
    ) = load_data_and_create_generators(
        skip_non_cached=skip_non_cached,
        model=model
    )
    i = 0
    for sample in train_generator:
        print(
            sample[2].shape
        )  # Just make sure that we can get a single sample out of the dry-run
        i = i + 1
        if i > 64:
            break
