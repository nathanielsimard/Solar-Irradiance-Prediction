import logging

from src.train import data


def main():
    train_dataset, valid_dataset, test_dataset = data.load_data()
    for image, target in train_dataset.batch(64):
        print(image.shape)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )

    main()
