import os
from os.path import dirname

import jsonpickle
import keras
import sklearn.neighbors as base_model
import tensorflow as tf
from keras import Model
from keras import layers
from keras import metrics
from keras import optimizers

from nn.dataset_provider import DatasetProvider
from nn.encoder import EncoderProvider
from nn.simclr_model import SimCLRModel

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"TensorFlow is using GPU: {physical_devices[0]}")
else:
    print("No GPU devices found. TensorFlow will run on CPU.")


class DistanceLayer(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SimCLRModelFramework:

    def handle_model(self) -> Model:
        train_dataset, val_dataset = DatasetProvider().prepare_dataset()
        model = self.__create_and_train_model(train_dataset, val_dataset)
        model.save("encoder")
        self.calculate_embeddings(model, train_dataset + val_dataset)
        return model

    def calculate_similarity(self, anchor_embedding, second_embedding):
        cosine_similarity = metrics.CosineSimilarity()
        return cosine_similarity(anchor_embedding, second_embedding).numpy()

    def calculate_embeddings(self, encoder, dataset):
        track_names = os.listdir("../data/music_wav")
        result = {}
        i = 0
        prefetch = dataset.batch(EncoderProvider.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        while i < len(track_names):
            dataset_iter = iter(prefetch)
            for _ in range(94):
                if i == len(track_names):
                    break
                anchor, positive, negative = next(dataset_iter)
                for vector in encoder(anchor):
                    track_name = track_names[i]
                    vector = vector.numpy()
                    result[track_name[:-4]] = vector
                    i += 1
                    if i == len(track_names):
                        break

        with open("embeddings.json", "w") as text_file:
            text_file.write(jsonpickle.encode(result))

    def get_model(self):
        return base_model.NearestNeighbors()

    @staticmethod
    def __create_and_train_model(train_dataset, val_dataset):
        simclr_network = SimCLRModelFramework.create_simclr_network()
        simclr_network = SimCLRModel(simclr_network)
        simclr_network.compile(optimizer=optimizers.Adam(0.0001))
        simclr_network.fit(train_dataset, epochs=10, validation_data=val_dataset)
        return simclr_network

    @staticmethod
    def create_simclr_network() -> Model:
        saved_model_name = next((
            directory for directory in os.listdir(dirname(__file__))
            if directory.startswith("siamese")),
            None
        )
        if saved_model_name:
            return keras.models.load_model(saved_model_name)

        encoder = EncoderProvider.get_network()

        anchor_input = layers.Input(name="anchor", shape=EncoderProvider.target_shape,
                                    batch_size=EncoderProvider.batch_size)
        positive_input = layers.Input(name="positive", shape=EncoderProvider.target_shape,
                                      batch_size=EncoderProvider.batch_size)
        negative_input = layers.Input(name="negative", shape=EncoderProvider.target_shape,
                                      batch_size=EncoderProvider.batch_size)

        distances = DistanceLayer()(
            encoder(anchor_input),
            encoder(positive_input),
            encoder(negative_input),
        )

        simclr_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

        return simclr_network
