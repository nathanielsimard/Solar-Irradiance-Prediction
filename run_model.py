import argparse
import random

import tensorflow as tf
from tensorflow.keras import optimizers

from src import dry_run, env
from src.model import (
    autoencoder,
    base,
    clearsky,
    conv2d,
    conv3d,
    conv3d_lm,
    embed_conv3d,
    gru,
)
from src.session import Session

MODELS = {
    autoencoder.NAME_AUTOENCODER: autoencoder.Autoencoder,
    conv2d.NAME: conv2d.CNN2D,
    conv2d.NAME_CLEARSKY: conv2d.CNN2DClearsky,
    conv2d.NAME_CLEARSKY_8x8: conv2d.CNN2DClearsky_8x8,
    conv3d.NAME: conv3d.CNN3D,
    embed_conv3d.NAME: embed_conv3d.Conv3D,
    conv3d_lm.NAME: conv3d_lm.Conv3D,
    clearsky.NAME: clearsky.ClearskyMLP,
    gru.NAME: gru.GRU,
}


def create_model(model_name: str) -> base.Model:
    """Create the model from its name."""
    try:
        return MODELS[model_name]()
    except KeyError:
        raise ValueError(
            f"Bad model name, {model_name} do not exist.\n"
            + f"Available models are {list(MODELS.keys())}"
        )


def parse_args():
    """Parse the user's arguments.

    The default arguments are to be used in order to reproduce
    the original experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_file",
        help="Tensorflow caching apply after model's preprocessing."
        + "Note that this cache must be used only for a model with a specific configuration."
        + "It must not be shared between models or the same model with different configuration.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run_local", help="Enable training with relative paths", action="store_true"
    )
    parser.add_argument(
        "--epochs", help="Number of epoch to train", default=25, type=int
    )
    parser.add_argument(
        "--test",
        help="Test a trained model on the test set. The value must be the model's checkpoint",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--train", help="Train a model.", action="store_true",
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

    parser.add_argument(
        "--seed", help="Seed for the experiment", default=1234, type=int
    )

    parser.add_argument(
        "--random_seed",
        help="Will overide the default seed and use a random one",
        action="store_true",
    )

    parser.add_argument(
        "--no_checkpoint", help="Will not save any checkpoints", action="store_true",
    )

    parser.add_argument("--lr", help="Learning rate", default=0.001, type=float)

    parser.add_argument(
        "--model",
        help=f"Name of the model to train, available models are:\n{list(MODELS.keys())}",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", help="Batch size", default=128, type=int)
    return parser.parse_args()


def run(args):
    """Run the model with RMSE Loss.

    It can train or test with different datasets.
    """
    env.run_local = args.run_local

    if not args.random_seed:
        random.seed(args.seed)
        tf.random.set_seed(args.seed)

    if args.dry_run:
        dry_run.run(args.enable_tf_caching, args.skip_non_cached)

    model = create_model(args.model)

    session = Session(
        model=model, batch_size=args.batch_size, skip_non_cached=args.skip_non_cached,
    )

    if args.train:
        optimizer = optimizers.Adam(args.lr)
        session.train(
            optimizer=optimizer,
            cache_file=args.cache_file,
            enable_checkpoint=not args.no_checkpoint,
            epochs=args.epochs,
        )

    if args.test is not None:
        session.test(args.test)


if __name__ == "__main__":
    args = parse_args()
    run(args)
