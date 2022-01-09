import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import build_model
from postprocessing import calculate_delivery_date
from preprocessing import preprocess

TRAINING = False
TEST_FILE_NAME = "quiz-hamed"
MODEL_WEIGHT_PATH = './data/model_weights.hdf5'


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


######################################## Preprocessing Data Phase #######################################
# Load and preprocess dataset
X, x_quiz, y, w = preprocess("./data/train.h5", "./data/" + TEST_FILE_NAME + ".h5")

# Convert data to ndarrays
X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')
w = np.asarray(w / w.mean()).astype('float32')  # set average of weights to one

# Scale features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
x_quiz = scaler.transform(x_quiz)

######################################## Training Phase #######################################
model = build_model()

earlyStopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', min_delta=1e-5)
mcp_save = ModelCheckpoint(MODEL_WEIGHT_PATH, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1, min_delta=1e-4, mode='min')
# tensor_board = TensorBoard(log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)


if TRAINING:
    model.compile(metrics=[maebay], loss="mae", optimizer="adam")
    history = model.fit(X, y, verbose=2, epochs=100, batch_size=2048, sample_weight=w, validation_split=0.05,
                        callbacks=[reduce_lr_loss, mcp_save, earlyStopping])


model = load_model(MODEL_WEIGHT_PATH, custom_objects={"maebay": maebay})

######################################## Prediction Phase #######################################
predictions = model.predict(x_quiz)
np.savetxt("./output/quiz_result.csv", predictions, delimiter=",")
calculate_delivery_date("./data/" + TEST_FILE_NAME + ".tsv")
