# Configuring logger.
from src import logging  # noqa: F401
from src.model import conv2d


def main():
    model = conv2d.create_model()
    conv2d.train(model)


if __name__ == "__main__":
    main()
