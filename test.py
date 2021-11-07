import numpy as np
import tensorflow.keras.backend as K


def maebay(y_true, y_pred):
    weight = (K.sign(y_true - y_pred) * 0.1) + 0.5
    eval = K.abs(y_true - y_pred)
    eval = weight * eval
    eval = K.mean(eval, axis=-1)
    return eval


a = np.array(K.constant([1, 4, 3]))
b = np.array(K.constant([2, 2, 3]))

print(mae(a, b))
