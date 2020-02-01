import numpy as np
from typing import Dict
from src.data.metadata import Coordinates
from src.data.dataloader import ImageReader

BND_COORDINATES = Coordinates(40.05192, -88.37309, 230)
BND_CENTER_POINT_PXL = (500, 300)


def coordinates_to_pixel(
    coordinates: Coordinates, image_path: str, offset: int
) -> Dict[str, int]:
    """Return the x and y pixel value in an image of a given coordinate."""
    # Latitude: 650
    image_reader_lat = ImageReader(channels=["lat"])
    image_lat = image_reader_lat.read(image_path, offset)

    # Longitude: 1500
    image_reader_lon = ImageReader(channels=["lon"])
    image_lon = image_reader_lon.read(image_path, offset)

    pixel_x = np.argmin(np.abs(image_lat - coordinates.latitude))
    pixel_y = np.argmin(np.abs(image_lon - coordinates.longitude))

    return {"x": pixel_x, "y": pixel_y}


def crop_around_pixel(
    image: np.array, center_pixel: Dict[str, int], output_size=(64, 64)
) -> np.array:
    """Crop a given image around the  center pixel."""
    start_x = center_pixel["x"] - (output_size[0] // 2)
    start_y = center_pixel["y"] - (output_size[1] // 2)

    end_x = start_x + output_size[0]
    end_y = start_y + output_size[1]

    return image[start_x:end_x, start_y:end_y]
