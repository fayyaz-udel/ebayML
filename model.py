import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


EPOCHS = 50
LR = 0.001
BATCH = 128


def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss="mean_squared_error",
                  optimizer=tf.keras.optimizers.Adam(LR))
    print(model.summary())
    return model


def train_model(dnn_model, train_features, train_labels):
    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.1,
        verbose=2, epochs=EPOCHS, batch_size=BATCH)

    return history
