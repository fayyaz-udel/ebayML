import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing

from model import build_and_compile_model, train_model
from preprocessing import preprocess

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

X, x_quiz, y = preprocess("./data/train.tsv", "./data/quiz.tsv", 10000000, False)

X = np.asarray(X).astype('float32')
x_quiz = np.asarray(x_quiz).astype('float32')

#normalizer = preprocessing.Normalization(axis=-1)
#normalizer.adapt(np.array(X))
#model = build_and_compile_model(normalizer)
#history = train_model(model, X, y)
#logging.info("start saving result for quiz set")
#np.savetxt("./data/quiz_result.csv", model.predict(x_quiz), delimiter=",")


reg = GradientBoostingRegressor(verbose=2, max_depth=5)
reg.fit(X, y)
np.savetxt("./data/quiz_result.csv", reg.predict(x_quiz), delimiter=",")


logging.info("finished")




