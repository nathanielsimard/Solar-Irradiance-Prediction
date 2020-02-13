import abc

import tensorflow as tf

from src.data import dataloader

MODEL_BASE_DIR = "/project/cq-training-1/project1/teams/team10/models"


class Model(tf.keras.Model, abc.ABC):
    """All models will inherit from this class."""

    def __init__(self, title: str):
        super().__init__()
        self.title = title

    def save(self, instance: str):
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().save_weights(
            file_name, save_format="tf", overwrite=True,
        )

    def load(self, instance: str):
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().load_weights(file_name)

    @abc.abstractmethod
    def config(self, training=False) -> dataloader.Config:
        pass

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset
