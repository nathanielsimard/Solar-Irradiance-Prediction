import timeit

from src.data import image
from src.data.metadata import Coordinates

IMAGE_PATH = "tests/data/samples/2015.11.01.0800.h5"
COORDINATES = Coordinates(40.05192, -88.37309, 230)


class ImageReaderPerf(object):
    def __init__(self, channels, output_size):
        self.output_size = output_size
        self.image_reader = image.ImageReader(channels=channels)

    def run(self):
        self.image_reader.read(IMAGE_PATH, 8, COORDINATES, output_size=self.output_size)


def run():
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
    run()
