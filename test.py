import os

import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"TensorFlow is using GPU: {physical_devices[0]}")
    else:
        print("No GPU devices found. TensorFlow will run on CPU.")
