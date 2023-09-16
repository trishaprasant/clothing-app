import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

def create_model():
    data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    ])

    resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(28, 28),
    layers.Rescaling(1./255)
    ])

    preprocessing_layer = tf.keras.Sequential([
    data_augmentation,
    resize_and_rescale,
    ])

    model = Sequential()
    model.add(preprocessing_layer)

    model.add(Conv2D(64, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)))


    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    return model