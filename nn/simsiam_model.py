from os.path import join, dirname

import jsonpickle
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
# tensorflow 2.10.0 !!!!!!!!! to work with gpu
from pathlib import Path
import gdown

from keras import layers
from keras import api
from keras import losses
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet
import zipfile

from keras.callbacks import TensorBoard
import datetime
from numpy import shape

from keras.utils import plot_model

target_shape = (200, 200)
batch_size = 64

cache_dir = join(dirname(__file__), '../data/cache')
anchor_images_path = join(cache_dir, "left")
positive_images_path = join(cache_dir, "right")

# !gdown --id 1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34
# !gdown --id 1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW
# !unzip -oq left.zip -d $cache_dir
# !unzip -oq right.zip -d $cache_dir

# zf = zipfile.ZipFile('left.zip')
# zf.extractall(path=cache_dir)
# zf.close()
#
# zf = zipfile.ZipFile('right.zip')
# zf.extractall(path=cache_dir)
# zf.close()

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"TensorFlow is using GPU: {physical_devices[0]}")
else:
    print("No GPU devices found. TensorFlow will run on CPU.")


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


# We need to make sure both the anchor and positive images are loaded in
# sorted order so we can match them together.
anchor_images = sorted(
    [join(anchor_images_path, f) for f in os.listdir(anchor_images_path)]
)

positive_images = sorted(
    [join(positive_images_path, f) for f in os.listdir(positive_images_path)]
)

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

# To generate the list of negative images, let's randomize the list of
# available images and concatenate them together.
rng = np.random.RandomState(seed=42)
rng.shuffle(anchor_images)
rng.shuffle(positive_images)

negative_images = anchor_images + positive_images
np.random.RandomState(seed=32).shuffle(negative_images)

negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
negative_dataset = negative_dataset.shuffle(buffer_size=4096)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Let's now split our dataset in train and validation.
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


def visualize(anchor, positive, negative):

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])


base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False,
)

flatten = layers.Flatten()(base_cnn.output)

dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)

dense2 = layers.Dense(512, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)

dense3 = layers.Dense(256, activation="relu")(dense2)
dense3 = layers.BatchNormalization()(dense3)

dense4 = layers.Dense(512, activation="relu")(dense3)
dense4 = layers.BatchNormalization()(dense4)

dense5 = layers.Dense(512, activation="relu")(dense4)
dense5 = layers.BatchNormalization()(dense5)

output = layers.Dense(256)(dense3)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

saved_model_name = next((
    directory for directory in os.listdir(dirname(__file__))
    if directory.startswith("embedding")),
    None
)
if saved_model_name:
    embedding = keras.models.load_model(saved_model_name)

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)


class SiameseModel(Model):

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]


def visualize_training():
    log_dir = "../data/tf_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return [TensorBoard(log_dir=log_dir,
                        histogram_freq=0,
                        write_graph=False,
                        write_images=False,
                        update_freq='epoch',
                        profile_batch=2,
                        embeddings_freq=0)]


def create_siamese_network() -> Model:
    saved_model_name = next((
        directory for directory in os.listdir(dirname(__file__))
        if directory.startswith("siamese")),
        None
    )
    if saved_model_name:
        return keras.models.load_model(saved_model_name)

    return siamese_network


siamese_model = SiameseModel(create_siamese_network())
# siamese_model.compile(optimizer=optimizers.Adam(0.0001))
# siamese_model.fit(train_dataset, epochs=1, validation_data=val_dataset, callbacks=visualize_training())

# siamese_network.save("siamese")
# embedding.save("embedding")

sample = next(iter(train_dataset))

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
)
cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())
# 6016 elemtns 94 iters
track_names = os.listdir("../data/music_wav")
result = {}
i = 0
prefetch = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
while i < len(track_names):
    dataset_iter = iter(prefetch)
    for _ in range(94):
        if i == len(track_names):
            break
        anchor, positive, negative = next(dataset_iter)
        for vector in embedding(resnet.preprocess_input(anchor)):
            track_name = track_names[i]
            vector = vector.numpy()
            result[track_name[:-4]] = vector
            i += 1
            if i == len(track_names):
                break

print(len(result))
print(type(result["$NOT - BERETTA (feat. Wifisfuneral)"]))
print(result["$NOT - BERETTA (feat. Wifisfuneral)"])
with open("embeddings.json", "w") as text_file:
    text_file.write(jsonpickle.encode(result))
