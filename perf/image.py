import timeit
from datetime import datetime

from src.data import dataloader, image
from src.data.metadata import Coordinates, Metadata, Station
from src.data.train import default_config

IMAGE_PATH = "tests/data/samples/2015.11.01.0800.h5"
COORDINATES = Coordinates(40.05192, -88.37309, 230)
STATION = Station.BND


class ImageReaderPerf(object):
    def __init__(self, channels, output_size):
        self.output_size = output_size
        self.image_reader = image.ImageReader(channels=channels)

    def run(self):
        self.image_reader.read(IMAGE_PATH, 8, COORDINATES, output_size=self.output_size)


class DatasetPerf(object):
    def __init__(self, num_images):
        self.num_images = num_images

        config = default_config()
        config.features = [dataloader.Feature.image]

        def gen():
            for _ in range(num_images):
                yield Metadata(IMAGE_PATH, "8bits", 10, datetime.now(), COORDINATES)

        self.dataset = dataloader.create_dataset(lambda: gen(), config=config)

    def run(self):
        for i, m in enumerate(self.dataset):
            if i % 100 == 0:
                print(f"Loaded {i} images")


def run_dataset():
    print("--- Dataset Image Only Benchmark ---")
    num_images = 1000
    test = DatasetPerf(num_images)
    num_iter = 10
    result = timeit.timeit(test.run, number=num_iter)
    print(f"Read {(num_images*num_iter)/result} img/sec")


def run_image_loader():
    for channels, output_size in [
        (["ch1"], (32, 32)),
        (["ch1"], (64, 64)),
        (["ch1", "ch2", "ch3"], (64, 64)),
        (["ch1", "ch2", "ch3", "ch4", "ch6"], (64, 64)),
        (["ch1", "ch2", "ch3", "ch4", "ch6"], (128, 128)),
        (["ch1", "ch2", "ch3", "ch4", "ch6"], (256, 256)),
    ]:
        perf_test = ImageReaderPerf(channels, output_size)
        print("--- Image Reder Benchmark ---")
        number = 100
        result = timeit.timeit(perf_test.run, number=number)
        print(
            f"Image reading rate [{number/result:0.3f} img/s]\n"
            + f"output_size={output_size}\n"
            + f"channels={channels}\n"
        )


if __name__ == "__main__":
    run_dataset()
