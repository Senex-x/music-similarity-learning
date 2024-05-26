import keras
import keras_cv
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
import numpy

strength = [0.4, 0.4, 0.4, 0.1]
random_brightness = layers.RandomBrightness(0.8 * strength[0])
random_contrast = layers.RandomContrast((1 - 0.8 * strength[1], 1 + 0.8 * strength[1]))
random_saturation = keras_cv.layers.RandomSaturation(
    (0.5 - 0.8 * strength[2], 0.5 + 0.8 * strength[2])
)
random_hue = keras_cv.layers.RandomHue(0.2 * strength[3], [0,255])
grayscale = keras_cv.layers.Grayscale()

def color_drop(x):
    x = grayscale(x)
    x = tf.keras.backend.tile(x, [1, 1, 3])
    return x

def color_jitter(x):
    x = random_brightness(x)
    x = random_contrast(x)
    x = random_saturation(x)
    x = random_hue(x)
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    return x

def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p: # here changed
        return func(x)
    else:
        return x

def custom_augment(image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = color_jitter(image)
    # image = random_apply(color_jitter, image, p=0.8)
    # image = random_apply(color_drop, image, p=0.2)
    return image


image = plt.imread("data/music_spectrograms/anchor/0000.png")

image = custom_augment(image)

plt.imshow(image)
plt.show()
plt.imshow(image.numpy())
plt.show()
plt.imshow(image.numpy().astype("int"))
plt.show()

print(numpy.core.shape(image))
print(type(image))
print(image[:1])

print(numpy.core.shape(image))
print(type(image.numpy()))
print(image[:1])
