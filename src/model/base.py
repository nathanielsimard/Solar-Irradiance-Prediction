import abc

import tensorflow as tf

from src.data import dataloader

MODEL_BASE_DIR = "models"


class Model(tf.keras.Model, abc.ABC):
    """All models will inherit from this class.

    Each model must supplie their configuration with what features they need.
    Each model has full control over the preprocessing apply on the data.
    """

    def __init__(self, title: str):
        """Name of the model."""
        super().__init__()
        self.title = title

    def save(self, instance: str):
        """Save the model weights."""
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().save_weights(
            file_name, save_format="tf", overwrite=True,
        )

    def load(self, instance: str):
        """Loading the model weights."""
        file_name = f"{MODEL_BASE_DIR}/{self.title}/{instance}"
        super().load_weights(file_name)

    @abc.abstractmethod
    def config(self) -> dataloader.DataloaderConfig:
        """Each model must have a config method."""
        pass

    def preprocess(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Models can apply a preprocessing pipeline.

        For example, normalization of features should be done here.
        """
        return dataset


def crop_image(image: tf.Tensor, crop_size):
    """Performs dynamic cropping of an image."""
    image_size_x = image.shape[0]
    image_size_y = image.shape[1]
    pixel = crop_size
    start_x = image_size_x // 2 - pixel // 2
    end_x = image_size_x // 2 + pixel // 2
    start_y = image_size_y // 2 - pixel // 2
    end_y = image_size_y // 2 + pixel // 2
    cropped_image = image[start_x:end_x, start_y:end_y, :]

    return cropped_image
