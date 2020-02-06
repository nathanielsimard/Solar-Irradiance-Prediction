# Configuring logger.
from src import logging  # noqa: F401
from src.model import conv2d


def main():
    conv2d.train(None)


if __name__ == "__main__":
    main()
