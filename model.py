import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

EPOCHS = 100
BATCH = 64


def build_and_compile_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss="mean_absolute_error",
                  optimizer=tf.keras.optimizers.Adam())
    return model


def train_model(dnn_model, train_features, train_labels):
    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.1,
        verbose=2, epochs=EPOCHS, batch_size=BATCH)

    return history
