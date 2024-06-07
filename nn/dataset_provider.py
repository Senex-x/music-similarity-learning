import os
from os.path import join, dirname
import numpy
import numpy as np
import tensorflow as tf


class DatasetProvider:

    def prepare_dataset(self):
        anchor_data_path = join(dirname(__file__), '../data/music_tokens')
        anchor_examples = sorted(
            [join(anchor_data_path, f) for f in os.listdir(anchor_data_path)]
        )
        positive_examples = sorted(
            [self.__augment_data_with_distortion(anchor) for anchor in anchor_examples]
        )
        sample_count = len(anchor_examples)

        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_examples)
        positive_dataset = tf.data.Dataset.from_tensor_slices(positive_examples)

        rng = np.random.RandomState(seed=42)
        rng.shuffle(anchor_examples)
        rng.shuffle(positive_examples)

        negative_images = anchor_examples + positive_examples
        np.random.RandomState(seed=32).shuffle(negative_images)

        negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
        negative_dataset = negative_dataset.shuffle(buffer_size=4096)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        dataset = dataset.shuffle(buffer_size=1024)

        train_dataset = dataset.take(round(sample_count * 0.8))
        val_dataset = dataset.skip(round(sample_count * 0.8))

        train_dataset = train_dataset.batch(64, drop_remainder=False)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = val_dataset.batch(64, drop_remainder=False)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    @staticmethod
    def __augment_data_with_distortion(matrix):
        noise = np.random.normal(0, 30, np.shape(matrix))
        return numpy.add(matrix, noise)
