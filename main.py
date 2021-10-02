import numpy as np
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

X, x_quiz, y = preprocess("./data/train.tsv", "./data/quiz.tsv", False)

y.to_csv("fsdfse.csv")
logging.info("20")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)
logging.info("21")
#LogisticRegression(random_state=0)  #
#reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3, verbose=2), verbose=2)
reg = GradientBoostingRegressor(verbose=2)
logging.info("22")
reg.fit(X_train, y_train)
logging.info("23")
np.savetxt("./data/quiz_result.csv", reg.predict(x_quiz), delimiter=",")
logging.info("finished")
