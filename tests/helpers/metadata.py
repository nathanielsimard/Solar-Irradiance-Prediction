from datetime import datetime

from src.data.dataloader import DataLoader, ImageReader
from src.data.metadata import Coordinates, Metadata

BND_COORDINATES = Coordinates(40.05192, -88.37309, 230)

IMAGE_PATH_1 = "tests/data/samples/2015.11.01.0800.h5"
IMAGE_PATH_2 = "tests/data/samples/2015.11.02.0800.h5"
IMAGE_SIZE = (650, 1500)


def create_dataset(channels):
    dataloader = DataLoader(ImageReader(channels=channels))
    return dataloader.create_dataset(_metadata_iterable())


def _metadata_iterable():
    for image_path in [IMAGE_PATH_1, IMAGE_PATH_2]:
        yield Metadata(
            image_path,
            "8bits",
            6,
            datetime.now(),
            BND_COORDINATES,
            target_ghi=100,
            target_ghi_1h=100,
            target_ghi_3h=100,
            target_ghi_6h=100,
        )
