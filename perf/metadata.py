import timeit

from src.data.metadata import Coordinates, MetadataLoader, Station

CATALOG_PATH = "tests/data/samples/catalog-test.pkl"
COORDINATES = Coordinates(40.05192, -88.37309, 230)
STATION = Station.BND


class MetadataPerf(object):
    def __init__(self):
        self.loader = MetadataLoader(CATALOG_PATH)

    def run(self):
        metadata = self.loader.load(STATION, COORDINATES, skip_missing=False)
        i = 0
        for m in metadata:
            i += 1
        print(i)


def run():
    perf_test = MetadataPerf()
    print("--- Image Reder Benchmark ---")
    number = 10
    num_metadata = 2066
    result = timeit.timeit(perf_test.run, number=number)
    print(f"Load metadata [{number*num_metadata/result:0.3f} medatada/s]\n")


if __name__ == "__main__":
    run()
