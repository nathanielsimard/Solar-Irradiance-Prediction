from dataclasses import dataclass

import tensorflow as tf

from src.data.metadata import Coordinates

BND_COORDINATES = Coordinates(40.05192, -88.37309, 230)
BND_CENTER_POINT_PXL = (500, 300)


def center_station_coordinates(
    dataset: tf.data.Dataset, coordinates=BND_COORDINATES, output_size=(64, 64),
) -> tf.data.Dataset:
    return dataset.map(
        lambda data, target: _center_image(data, output_size, BND_CENTER_POINT_PXL)
    )


def _center_image(image, output_size, center_point_px):
    height = output_size[0]
    width = output_size[1]

    height_2 = int(height / 2)
    width_2 = int(width / 2)

    offset_height = center_point_px[1] - height_2
    offset_width = center_point_px[0] - width_2

    return tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, height, width
    )
