import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from postprocessing import calculate_delivery_date
from preprocessing import preprocess
import tensorflow.keras.backend as K


def maebay(y_true, y_pred):
    tf.cast(y_true, tf.float32)
    tf.cast(y_pred, tf.float32)
    weight = K.sign(y_true - y_pred)
    weight = weight * 0.1
    weight = weight + 0.5
    eval = K.abs(y_true - y_pred)
    eval = weight * eval
    eval = K.mean(eval, axis=-1)
    return eval


X, x_quiz, y, w = preprocess("./data/train.h5", "./data/quiz.h5")
X = X[y < 20]
y = y[y < 20]
print(X.info())

X = np.asarray(X).astype('float32')
x_quiz = np.asarray(x_quiz).astype('float32')
y = np.asarray(y).astype('float32')
w = np.asarray(w).astype('float32')

print("weight shape: ")
print(w.shape)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
x_quiz = scaler.transform(x_quiz)

##### Training Phase ####

model = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='linear')])

earlyStopping = EarlyStopping(monitor='val_loss', patience=10 , verbose=1, mode='min', min_delta=1e-5)
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_delta=1e-4, mode='min')
#tensor_board = TensorBoard(log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)


model.compile(metrics=[maebay], loss="mae", optimizer="adam")
history = model.fit(X, y, validation_split=0.05, verbose=2, epochs=100, batch_size=128, sample_weight=w, callbacks=[reduce_lr_loss, mcp_save, earlyStopping])

np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")

calculate_delivery_date()
logging.info("finished")