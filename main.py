import logging

import xgboost
import xgboost as xgb

import numpy as np
from keras.losses import MSE, MAE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from postprocessing import calculate_delivery_date
from preprocessing import preprocess

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

X, x_quiz, y = preprocess("./data/train_w_zip.tsv", "./data/quiz_w_zip.tsv", False)
X = np.asarray(X).astype('float32')
x_quiz = np.asarray(x_quiz).astype('float32')

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.0001)
##### Training Phase ####
model = XGBRegressor(n_estimators=1000, max_depth=8, verbosity=2, tree_method='gpu_hist')

model.fit(train_X, train_y, eval_set=[(test_X, test_y)],
        eval_metric='mae',
        verbose=True)
pred = model.predict(test_X)
pred_test = model.predict(train_X)
test_mse = MSE(test_y, pred)
test_mae = MAE(test_y, pred)/2
print("TEST MSE : % f" %(test_mse))
print("TEST MAE : % f" %(test_mae))

train_mse = MSE(train_y, pred_test)
train_mae = MAE(train_y, pred_test)/2
print("TRAIN MSE : % f" %(train_mse))
print("TRAIN MAE : % f" %(train_mae))
np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")



# normalizer = preprocessing.Normalization(axis=-1)
# normalizer.adapt(np.array(X))
# model = build_and_compile_model(normalizer)
# history = train_model(model, X, y)
# logging.info("start saving result for quiz set")
# np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")
#########################


calculate_delivery_date()
logging.info("finished")

# dtrain = xgb.DMatrix(X, label=y)
# dtest = xgb.DMatrix(x_quiz)
# param = {}
# param['tree_method'] = 'gpu_hist'
# param['verbosity'] = 2
# bst = xgb.train(param, dtrain, verbose_eval=True, num_boost_round=100)
# np.savetxt("./data/quiz_result.csv", bst.predict(dtest), delimiter=",")
