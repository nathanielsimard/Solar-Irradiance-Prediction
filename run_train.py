import argparse

from tensorflow.keras import losses, optimizers

from src import dry_run, env
from src.model import cnn_plus_lstm
from src.training import Training


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

    parser.add_argument("--lr", help="Learning rate", default=0.0001, type=float)

    parser.add_argument("--model", help="Name of the model to train", default="CNN2D")
    parser.add_argument("--batch_size", help="Batch size", default=128, type=int)
    args = parser.parse_args()
    env.run_local = args.run_local

    if args.dry_run:
        dry_run.run(args.enable_tf_caching, args.skip_non_cached)
        return

    model = cnn_plus_lstm.CNNLSTM()
    optimizer = optimizers.Adam(0.01)
    loss_obj = losses.MeanSquaredError()

    def rmse(pred, target):
        return loss_obj(pred, target) ** 0.5

    training_session = Training(optimizer=optimizer, model=model, loss_fn=rmse)
    training_session.run(
        enable_tf_caching=args.enable_tf_caching,
        skip_non_cached=args.skip_non_cached,
        enable_checkpoint=not args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
