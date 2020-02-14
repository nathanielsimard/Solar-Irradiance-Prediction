import argparse

from tensorflow.keras import losses, optimizers

from src import dry_run, env
from src.model import conv3d
from src.training import SupervisedTraining


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
        "--no_checkpoint", help="Will not save any checkpoints", action="store_true",
    )
    args = parser.parse_args()
    env.run_local = args.run_local

    if args.dry_run:
        dry_run.run(args.enable_tf_caching, args.skip_non_cached)
        return

    model = conv3d.Conv3D()
    optimizer = optimizers.Adam(0.001)
    loss_obj = losses.MeanSquaredError()

    def rmse(pred, target):
        return loss_obj(pred, target) ** 0.5

    training_session = SupervisedTraining(
        optimizer=optimizer, model=model, loss_fn=rmse
    )
    training_session.run(
        enable_tf_caching=args.enable_tf_caching,
        skip_non_cached=args.skip_non_cached,
        enable_checkpoint=not args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
