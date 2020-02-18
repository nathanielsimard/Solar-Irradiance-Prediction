import abc

import tensorflow as tf

from src.data import dataloader

MODEL_BASE_DIR = "/home/project1/models"


class Model(tf.keras.Model, abc.ABC):
    """All models will inherit from this class."""

    def __init__(self, title: str):
        """Name of the model."""
        super().__init__()
        self.title = title

    def save(self, instance: str):
        """Saving the model."""
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().save_weights(
            file_name, save_format="tf", overwrite=True,
        )

    def load(self, instance: str):
        """Loading the model."""
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().load_weights(file_name)

    @abc.abstractmethod
    def config(self, training=False) -> dataloader.DataloaderConfig:
        """Each model can have a config method."""
        pass

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Different models can apply a preprocessing pipeline."""
        return dataset
