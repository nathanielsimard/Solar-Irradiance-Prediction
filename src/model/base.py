import abc

import tensorflow as tf

from src.data import dataloader

MODEL_BASE_DIR = "/project/cq-training-1/project1/teams/team10/models"


class Model(tf.keras.Model, abc.ABC):
    def __init__(self, name: str):
        self.name = name

    def save(self, instance: str):
        file_name = f"{MODEL_BASE_DIR}/{self.name}/{instance}"
        super().save(
            file_name, save_format="tf", overwrite=True,
        )

    def load(self, instance: str):
        file_name = f"{MODEL_BASE_DIR}/{self.name}/{instance}"
        return super().load(file_name)

    @abc.abstractmethod
    def config(self, training=False) -> dataloader.Config:
        pass

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset
