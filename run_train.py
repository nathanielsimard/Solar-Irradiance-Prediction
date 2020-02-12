from src.model import conv2d
from src import env
import argparse
import logging


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable_tf_caching", help="Enable tensorflow caching.", action="store_true"
    )
    parser.add_argument(
        "--run_local", help="Enable training with relative paths", action="store_true"
    )
    parser.add_argument(
        "--dry_run",
        help="No training, no tensorflow, just the generator",
        action="store_true",
    )
    parser.add_argument(
        "--skip_non_cached",
        help="Skip images which are not already cached in the image reader",
        action="store_true",
    )
    args = parser.parse_args()
    env.run_local = args.run_local
    model = conv2d.create_model()
    conv2d.logger.setLevel(logging.DEBUG)
    conv2d.train(
        model,
        enable_tf_caching=args.enable_tf_caching,
        dry_run=args.dry_run,
        skip_non_cached=args.skip_non_cached,
    )


if __name__ == "__main__":
    main()
