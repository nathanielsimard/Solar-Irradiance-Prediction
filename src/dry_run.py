from src import logging

# from src.data.train import load_data_and_create_generators

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
    for sample in train_generator:
        print(
            sample
        )  # Just make sure that we can get a single sample out of the dry-run
        break
