from src.data import dataloader as dl
from src.data import metadata as meta
from tf.keras.Model import Sequential
from tf.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from tf.keras.activations import relu
from tf.keras.optimizers import SGD


def main():
    mdl = meta.MetadataLoader(
        "/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl"
    )
    md = mdl.load(
        meta.Station.BND,
        [40.05192, -88.37309, 230],
        night_time=False,
        skip_missing=True,
    )
    config = {}
    config["SKIP_MISSING"] = True
    (train_md, valid_md) = meta.train_valid_split(md)
    train_set = dl.create_dataset(train_md, config)
    valid_set = dl.create_dataset(valid_md, config)
    input_shape = (64, 64, 5)
    model = Sequential(
        [
            Conv2D(32, kernel_size=(8, 8), input_shape=input_shape),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(8, 8)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128),
            Activation("relu"),
            Dense(4),
        ]
    )
    print(model.summary())

    optimizer = SGD(0.0001)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mse"])
    historic = model.fit(train_set, validation_data=valid_set, epochs=10, batch_size=32)
    print(historic)


if __name__ == "__main__":
    main()
