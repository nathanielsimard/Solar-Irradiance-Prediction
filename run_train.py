import logging

from src import train


def main():
    train.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )
    main()
