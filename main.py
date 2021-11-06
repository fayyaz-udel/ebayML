import logging

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from postprocessing import calculate_delivery_date
from preprocessing import preprocess

X, x_quiz, y = preprocess("./data/train.h5", "./data/quiz.h5")
print(X.info())
X = np.asarray(X)  # .astype('float32')
x_quiz = np.asarray(x_quiz)  # .astype('float32')
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)
##### Training Phase ####
model = CatBoostRegressor(one_hot_max_size=100, n_estimators=2000, loss_function="MAE",
                          depth=16, logging_level='info', task_type='GPU')

model.fit(train_X, train_y, cat_features=[13, 14, 15], eval_set=(test_X, test_y), logging_level='Verbose')


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.001)

model = CatBoostRegressor(one_hot_max_size=100, n_estimators=2000, loss_function="MAE",
                          depth=16, logging_level='info', task_type='GPU')

model.fit(train_X, train_y, cat_features=[13, 14, 15], eval_set=(test_X, test_y), logging_level='Verbose')

pred = model.predict(test_X)
pred_test = model.predict(train_X)
test_mae = mean_absolute_error(test_y, pred) / 2
print("TEST MAE : % f" % (test_mae))
train_mae = mean_absolute_error(train_y, pred_test) / 2
print("TRAIN MAE : % f" % (train_mae))

np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")

calculate_delivery_date()
logging.info("finished")
