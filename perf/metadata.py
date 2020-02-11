import timeit

from src.data import dataloader
from src.data.metadata import Coordinates, MetadataLoader, Station
from src.data.train import default_config

CATALOG_PATH = "tests/data/samples/catalog-test.pkl"
COORDINATES = Coordinates(40.05192, -88.37309, 230)
STATION = Station.BND


class MetadataPerf(object):
    def __init__(self):
        self.loader = MetadataLoader(CATALOG_PATH)

    def run(self):
        metadata = self.loader.load(STATION, COORDINATES, skip_missing=False)
        for i, m in enumerate(metadata):
            if i % 100 == 0:
                print(f"Loaded {i} metadata")


class DatasetPerf(object):
    def __init__(self):
        loader = MetadataLoader(CATALOG_PATH)

        config = default_config()
        config.error_strategy = dataloader.ErrorStrategy.ignore
        config.features = [dataloader.Feature.target_ghi]

        self.dataset = dataloader.create_dataset(
            lambda: loader.load(STATION, COORDINATES, skip_missing=False), config=config
        )

    def run(self):
        for i, m in enumerate(self.dataset):
            if i % 100 == 0:
                print(f"Loaded {i} targets")


def run_metadataloader():
    print("--- Metadata Loader Benchmark ---")
    perf_test = MetadataPerf()
    number = 10
    num_metadata = 2066
    result = timeit.timeit(perf_test.run, number=number)
    print(f"Iterate over metadata [{number*num_metadata/result:0.3f} medatada/s]\n")


def run_dataset():
    print("--- Dataset Target Only Benchmark ---")
    dataset_perf = DatasetPerf()
    number = 10
    num_target = 2066
    result = timeit.timeit(dataset_perf.run, number=number)
    print(f"Iterate over dataset [{number*num_target/result:0.3f} target/s]\n")


if __name__ == "__main__":
    run_dataset()
