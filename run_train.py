from src.model import conv2d
from src.data.training import Training
from tensorflow.keras import optimizers, losses
from src import env
import argparse


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

    model = conv2d.CNN2D()
    optimizer = optimizers.SGD(0.0001)
    loss_obj = losses.MeanSquaredError()
    training_session = Training(optimizer=optimizer, model=model, loss_fn=loss_obj)
    training_session.run(enable_tf_caching=args.enable_tf_caching,
                         dry_run=args.dry_run, skip_non_cached=args.skip_non_cached,)


if __name__ == "__main__":
    main()
