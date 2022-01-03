import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def preprocess(train_address, test_address):
    original_df, train_end_index = load_dataset(train_address, test_address)
    feature_df = create_empty_df()

    add_binary_feature(original_df, feature_df, "b2c_c2c", "B2C")
    add_int_feature(original_df, feature_df, "long1")
    add_int_feature(original_df, feature_df, "lat1")
    add_int_feature(original_df, feature_df, "long2")
    add_int_feature(original_df, feature_df, "lat2")
    # add_int_feature(original_df, feature_df, "declared_handling_days")
    add_int_feature(original_df, feature_df, "shipping_fee")
    add_int_feature(original_df, feature_df, "distance")
    add_int_feature(original_df, feature_df, "carrier_min_estimate")
    add_int_feature(original_df, feature_df, "carrier_max_estimate")
    add_int_feature(original_df, feature_df, "item_price")
    add_int_feature(original_df, feature_df, "quantity")
    add_int_feature(original_df, feature_df, "weight")
    # add_categorical_feature(original_df, feature_df, "seller_id")
    add_int_feature(original_df, feature_df, "shipment_method_id")
    add_int_feature(original_df, feature_df, "category_id")
    original_df["package_size"] = original_df["package_size"].replace(
        {"NONE": 0, "LETTER": 1, "PACKAGE_THICK_ENVELOPE": 2, "LARGE_ENVELOPE": 3, "LARGE_PACKAGE": 4,
         "VERY_LARGE_PACKAGE": 4, "EXTRA_LARGE_PACKAGE": 4})
    add_int_feature(original_df, feature_df, "package_size")
    add_datetime_feature(original_df, feature_df, "acceptance_scan_timestamp")
    # add_datetime_feature(original_df, feature_df, "payment_datetime")

    label = add_label(original_df[:train_end_index], "acceptance_scan_timestamp")
    weight = add_weight(original_df[:train_end_index], "acceptance_scan_timestamp")

    X = feature_df[:train_end_index].fillna(0)
    x_quiz = feature_df[train_end_index:].fillna(0)

    X = X[label < 20]
    weight = weight[label < 20]
    label = label[label < 20]

    print(X.info())
    return X, x_quiz, label, weight


def save_df(df, name):
    df.to_csv(str(name) + ".csv")


def load_dataset(train_address, test_address):
    train_df = pd.read_hdf(train_address, 'df')
    train_df = shuffle(train_df)
    train_df = train_df.reset_index(drop=True)

    test_df = pd.read_hdf(test_address, 'df')
    train_df = clean_dataset(train_df)
    return pd.concat([train_df, test_df]), train_df.shape[0]


def clean_dataset(df):
    df = df[df["shipping_fee"] >= 0]
    #  df = df[df["package_size"] != "NONE"]
    df = df[df["carrier_min_estimate"] >= 0]
    df = df[df["carrier_max_estimate"] >= 0]
    df = df[df["distance"] >= 0]
    df = df[
        (pd.to_datetime(df["delivery_date"], infer_datetime_format=True) - pd.to_datetime(
            df["acceptance_scan_timestamp"].str.slice(0, 10), infer_datetime_format=True)).dt.days > 0]
    df = df[
        (pd.to_datetime(df["acceptance_scan_timestamp"].str.slice(0, 10), infer_datetime_format=True) - pd.to_datetime(
            df["payment_datetime"].str.slice(0, 10), infer_datetime_format=True)).dt.days >= 0]

    return df


def add_weight(original_df, time_feature_name):
    date_time = pd.to_datetime(original_df[time_feature_name].str.slice(0, 19), infer_datetime_format=True)
    return (date_time.dt.year - 2017.5) + (
                (date_time.dt.month - 1) * 0.0909090)  # + (((date_time.dt.month - 10.1).apply(np.sign) + 1)/3)


def add_label(original_df, start_point):
    start = pd.to_datetime(original_df[start_point].str.slice(0, 10), infer_datetime_format=True)
    end = pd.to_datetime(original_df["delivery_date"], infer_datetime_format=True)
    delta = (end - start).dt.days
    return delta


def create_empty_df():
    return pd.DataFrame()


def add_categorical_feature(original_df, feature_df, feature_name):
    dummies = pd.get_dummies(original_df[feature_name], prefix=feature_name)
    for col in dummies:
        feature_df[col] = dummies[col]
    return feature_df


def add_int_feature(original_df, feature_df, feature_name):
    feature_df[feature_name] = original_df[feature_name]

    if feature_name == "weight":
        # 2 kg --> 2.204 lbs
        feature_df[feature_name] = original_df[feature_name] * original_df["weight_units"].replace(2, 2.20462)


def add_binary_feature(original_df, feature_df, feature_name, one_value):
    feature_df[feature_name] = original_df[feature_name] == one_value


def add_datetime_feature(original_df, feature_df, feature_name):
    date_time = pd.to_datetime(original_df[feature_name].str.slice(0, 19), infer_datetime_format=True)
    feature_df[str(feature_name) + '_hour'] = date_time.dt.hour
    feature_df[str(feature_name) + '_day_week'] = date_time.dt.dayofweek
    feature_df[str(feature_name) + '_day'] = date_time.dt.day
    feature_df[str(feature_name) + '_month'] = date_time.dt.month

    feature_df[str(feature_name) + '_hour_sin'] = np.sin(date_time.dt.hour * (2. * np.pi / 24))
    feature_df[str(feature_name) + '_hour_cos'] = np.cos(date_time.dt.hour * (2. * np.pi / 24))

    feature_df[str(feature_name) + '_day_week_sin'] = np.sin(date_time.dt.dayofweek * (2. * np.pi / 7))
    feature_df[str(feature_name) + '_day_week_cos'] = np.cos(date_time.dt.dayofweek * (2. * np.pi / 7))

    feature_df[str(feature_name) + '_day_sin'] = np.sin((date_time.dt.day - 1) * (2. * np.pi / 31))
    feature_df[str(feature_name) + '_day_cos'] = np.cos((date_time.dt.day - 1) * (2. * np.pi / 31))

    feature_df[str(feature_name) + '_month_sin'] = np.sin((date_time.dt.month - 1) * (2. * np.pi / 12))
    feature_df[str(feature_name) + '_month_cos'] = np.cos((date_time.dt.month - 1) * (2. * np.pi / 12))
