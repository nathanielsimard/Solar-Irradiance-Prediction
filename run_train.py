import argparse

import tensorflow as tf
from tensorflow.keras import losses, optimizers

from src import dry_run, env
from src.model import autoencoder, embed_conv3d, languagemodel
from src.training import Training


def language_model():
    """Language Model."""
    encoder = autoencoder.Encoder()
    encoder.load("3")
    return languagemodel.Gru(encoder)


def conv3d_with_embeds():
    encoder = autoencoder.Encoder()
    encoder.load("3")
    return embed_conv3d.Conv3D(encoder)


def main():
    """Executable."""
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

    parser.add_argument("--lr", help="Learning rate", default=0.0001, type=float)

    parser.add_argument("--model", help="Name of the model to train", default="CNN2D")
    parser.add_argument("--batch_size", help="Batch size", default=128, type=int)
    args = parser.parse_args()
    env.run_local = args.run_local

    if not args.random_seed:
        tf.random.set_seed(args.seed)

    if args.dry_run:
        dry_run.run(args.enable_tf_caching, args.skip_non_cached)
        return

    optimizer = optimizers.Adam(0.001)
    loss_obj = losses.MeanSquaredError()

    model = conv3d_with_embeds()

    def rmse(pred, target):
        return loss_obj(pred, target) ** 0.5

    training_session = Training(optimizer=optimizer, model=model, loss_fn=rmse)
    training_session.run(
        enable_tf_caching=args.enable_tf_caching,
        skip_non_cached=args.skip_non_cached,
        enable_checkpoint=not args.no_checkpoint,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
