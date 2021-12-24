import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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


X, x_quiz, y = preprocess("./data/train.h5", "./data/quiz.h5")
X = X[y < 20]
y = y[y < 20]
print(X.info())


##### Training Phase ####
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.05)
model = CatBoostRegressor(one_hot_max_size=100, n_estimators=2000, loss_function="MAE", depth=16, logging_level='info', task_type='GPU')
model.fit(train_X, train_y, cat_features=[12, 13, 14], eval_set=(test_X, test_y), logging_level='Verbose')
np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")


# pred = model.predict(test_X)
# pred_test = model.predict(train_X)
# test_mae = mean_absolute_error(test_y, pred) / 2
# print("TEST MAE : % f" % (test_mae))
# train_mae = mean_absolute_error(train_y, pred_test) / 2
# print("TRAIN MAE : % f" % (train_mae))

calculate_delivery_date()
logging.info("finished")
