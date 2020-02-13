import abc

import tensorflow as tf

from src.data import dataloader

MODEL_BASE_DIR = "/project/cq-training-1/project1/teams/team10/models"


class Model(tf.keras.Model, abc.ABC):
    def __init__(self, title: str):
        super().__init__()
        self.title = title

    def save(self, instance: str):
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().save(
            file_name, save_format="tf", overwrite=True,
        )

    def load(self, instance: str):
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        return super().load(file_name)

    @abc.abstractmethod
    def config(self, training=False) -> dataloader.Config:
        pass

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset
