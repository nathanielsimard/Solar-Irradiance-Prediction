from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras import Model
from src import logging

# from tensorflow.keras.activations import relu

logger = logging.create_logger(__name__)


class CNN2D(Model):
    """Create Conv2D model."""

    def __init__(self):
        """Initialize the architecture."""
        super(CNN2D, self).__init__()
        input_shape = (64, 64, 5)
        self.conv1 = Conv2D(
            64, kernel_size=(5, 5), input_shape=input_shape, activation="relu"
        )
        self.mp1 = MaxPooling2D(
            pool_size=(2, 2)
        )  # not sure if it goes there, it does not in PyTorch...

        self.conv2 = Conv2D(128, kernel_size=(5, 5), activation="relu")
        self.mp2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3 = Conv2D(128, kernel_size=(3, 3), activation="relu")
        self.mp3 = MaxPooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.d1 = Dense(256, activation="relu")
        self.d2 = Dense(4)

    def __str__(self):
        """Name of the model."""
        return "Conv2D"

    def __call__(self, x, training: bool):
        """Performs the forward pass in the neural network.

        Can use a different pass with the optional training boolean if
        some operations need to be skipped at evaluation(e.g. Dropout)
        """
        x = self.conv1(x)
        x = self.mp1(x)  # Same here...
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
