import tensorflow as tf

from src.model import autoencoder, seq2seq


def seq2seq_gru_1():
    """Gru model based on the encoder."""
    tf.random.set_seed(1)
    encoder_weight = "3"
    num_images = 6
    time_interval_min = 60
    num_features = 16 * 16 * 32

    encoder = autoencoder.Encoder()
    encoder.load(encoder_weight)

    return seq2seq.Gru(
        encoder,
        num_images=num_images,
        time_interval_min=time_interval_min,
        num_features=num_features,
    )


def seq2seq_convlstm_1():
    """Conv LSTM model based on the encoder."""
    tf.random.set_seed(1)
    encoder_weight = "3"
    num_images = 6
    time_interval_min = 60
    num_channels = 5

    encoder = autoencoder.Encoder()
    encoder.load(encoder_weight)

    return seq2seq.ConvLSTM(
        encoder,
        num_images=num_images,
        time_interval_min=time_interval_min,
        num_channels=num_channels,
    )
