import logging

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers

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

X, x_quiz, y = preprocess("./data/train.h5", "./data/quiz.h5")
X = X[y < 20]
y = y[y < 20]
print(X.info())
X = np.asarray(X).astype('float32')
x_quiz = np.asarray(x_quiz).astype('float32')
y = np.asarray(y).astype('float32')
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1) TODO
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
x_quiz = scaler.transform(x_quiz)
##### Training Phase ####
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)])

model.compile(metrics=[maebay], loss="mae", optimizer="adam")
history = model.fit(X, y, validation_split=0.1, verbose=2, epochs=20, batch_size=128)

# model = CatBoostRegressor(one_hot_max_size=100, n_estimators=2000, loss_function="MAE", depth=16, logging_level='info', task_type='GPU')
# model.fit(train_X, train_y, cat_features=[9], eval_set=(test_X, test_y), logging_level='Verbose')


# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.001)
# model = CatBoostRegressor(one_hot_max_size=100, n_estimators=2000, loss_function="MAE", depth=16, logging_level='info', task_type='GPU')
# model.fit(train_X, train_y, cat_features=[13, 14, 15], eval_set=(test_X, test_y), logging_level='Verbose')

# pred = model.predict(test_X)
# pred_test = model.predict(train_X)
# test_mae = mean_absolute_error(test_y, pred) / 2
# print("TEST MAE : % f" % (test_mae))
# train_mae = mean_absolute_error(train_y, pred_test) / 2
# print("TRAIN MAE : % f" % (train_mae))

np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")

calculate_delivery_date()
logging.info("finished")
