import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from postprocessing import calculate_delivery_date
from preprocessing import preprocess

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

X, x_quiz, y = preprocess("./data/train_w_zip.tsv", "./data/quiz_w_zip.tsv", False)
X = np.asarray(X).astype('float32')
x_quiz = np.asarray(x_quiz).astype('float32')

##### Training Phase ####
reg = GradientBoostingRegressor(verbose=2, max_depth=10, n_estimators=25)
reg.fit(X, y)
np.savetxt("./output/quiz_result.csv", reg.predict(x_quiz), delimiter=",")
np.savetxt("./output/feature_importance.csv", reg.feature_importances_, delimiter=",")


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






