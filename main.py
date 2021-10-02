import logging

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing

from model import build_and_compile_model, train_model
from preprocessing import preprocess

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

X, x_quiz, y = preprocess("./data/train.tsv", "./data/quiz.tsv", 1000000, False)

X = np.asarray(X).astype('float32')


normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(X))

model = build_and_compile_model(normalizer)
history = train_model(model, X, y)

print(model.predict(x_quiz))

# np.savetxt("./data/quiz_result.csv", reg.predict(x_quiz), delimiter=",")
logging.info("finished")

# logging.info("20")

# logging.info("21")
# #LogisticRegression(random_state=0)  #
# #reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3, verbose=2), verbose=2)
# reg = GradientBoostingRegressor(verbose=2)
# logging.info("22")
# reg.fit(X_train, y_train)
# logging.info("23")
