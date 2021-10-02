import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.01))
    return model


def train_model(dnn_model, train_features, train_labels):
    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=2, epochs=100)

    return history
