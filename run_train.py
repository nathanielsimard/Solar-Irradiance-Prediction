import logging

from src.model import conv2d


def main():
    model = conv2d.create_model()
    conv2d.train(model)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )

    main()
