from preprocessing import preprocess
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

X, y = preprocess("./data/train.tsv", False)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)

reg.predict(X_test)

print(reg.score(X_test, y_test))

logging.info("finished")
