from matplotlib import pyplot as plt

from src.data.dataloader import Feature
from src.data.train import default_config, load_data


def plot3d(channel="ch6"):
    config = default_config()
    config.features = [Feature.image]
    config.channels = [channel]
    config.crop_size = (300, 300)
    config.time_interval_min = 60
    config.num_images = 7

    train, _, _ = load_data(config=config)
    time_wait = config.num_images * int(config.time_interval_min/15)

    for i, (image,) in enumerate(train.batch(1)):
        i += 1
        if i > time_wait:
            images = image
            break

    print(images.shape)
    for i, img in enumerate(images[0]):
        print(img.shape)
        plt.cla()
        plt.clf()
        plt.axis('off')
        imgg = img[:,:,0]
        print(imgg.shape)
        plt.imshow(imgg, cmap="gray")
        plt.savefig(f"image-3D/{i}.png")
