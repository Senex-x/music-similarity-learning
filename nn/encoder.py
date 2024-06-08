import keras
from keras import Model
from keras import layers


class EncoderProvider:
    target_shape = (128, 512)
    batch_size = 64
    encoder = None

    @staticmethod
    def get_network():
        if EncoderProvider.encoder:
            return EncoderProvider.encoder
        else:
            return EncoderProvider.__create_network()

    @staticmethod
    def __create_network():
        base_cnn = keras.models.Sequential()
        base_cnn.add(keras.layers.Input(shape=EncoderProvider.target_shape, batch_size=EncoderProvider.batch_size))
        flatten = layers.Flatten()(base_cnn.output)

        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        return Model(base_cnn.input, output, name="Encoder")
