import logging

import numpy as np
import pandas as pd


def preprocess(train_address, test_address):
    original_df, train_end_index = load_dataset(train_address, test_address)
    logging.info("train index: " + str(train_end_index))
    feature_df = create_empty_df()
    add_binary_feature(original_df, feature_df, "b2c_c2c", "B2C")
    add_int_feature(original_df, feature_df, "long1")
    add_int_feature(original_df, feature_df, "lat1")
    add_int_feature(original_df, feature_df, "long2")
    add_int_feature(original_df, feature_df, "lat2")
    add_int_feature(original_df, feature_df, "declared_handling_days")
    add_int_feature(original_df, feature_df, "shipping_fee")
    add_int_feature(original_df, feature_df, "distance")
    add_int_feature(original_df, feature_df, "carrier_min_estimate")
    add_int_feature(original_df, feature_df, "carrier_max_estimate")
    add_int_feature(original_df, feature_df, "item_price")
    add_int_feature(original_df, feature_df, "quantity")
    add_int_feature(original_df, feature_df, "weight")
    #add_categorical_feature(original_df, feature_df, "seller_id")
    add_categorical_feature(original_df, feature_df, "shipment_method_id")
    add_categorical_feature(original_df, feature_df, "category_id")
    add_categorical_feature(original_df, feature_df, "package_size")
    add_datetime_feature(original_df, feature_df, "acceptance_scan_timestamp")
    add_datetime_feature(original_df, feature_df, "payment_datetime")
    label = calculate_label(original_df[:train_end_index], "acceptance_scan_timestamp")

    logging.info(list(feature_df.columns))
    return feature_df[:train_end_index], feature_df[train_end_index:], label


def save_df(df, name):
    df.to_csv(str(name) + ".csv")


def load_dataset(train_address, test_address):
    train_df = pd.read_hdf(train_address, 'df')
    test_df = pd.read_hdf(test_address, 'df')
    train_df = clean_dataset(train_df)
    return pd.concat([train_df, test_df]), train_df.shape[0]


def clean_dataset(df):
    df = df[df["shipping_fee"] >= 0]
    df = df[df["package_size"] != "NONE"]
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


def calculate_label(original_df, start_point):
    start = pd.to_datetime(original_df[start_point].str.slice(0, 10), infer_datetime_format=True)
    end = pd.to_datetime(original_df["delivery_date"], infer_datetime_format=True)
    delta = (end - start).dt.days
    return delta


def create_empty_df():
    return pd.DataFrame()


def add_categorical_feature(original_df, feature_df, feature_name):
    feature_df[feature_name] = original_df[feature_name]
    # dummmies = pd.get_dummies(original_df[feature_name], prefix=feature_name)
    # for col in dummmies:
    #    feature_df[col] = dummmies[col]
    # return feature_df


def add_datetime_feature(original_df, feature_df, feature_name):
    date_time = pd.to_datetime(original_df[feature_name].str.slice(0, 19), infer_datetime_format=True)
    feature_df[str(feature_name) + 'day_week_sin'] = np.sin(date_time.dt.dayofweek * (2. * np.pi / 7))
    feature_df[str(feature_name) + 'day_week_cos'] = np.cos(date_time.dt.dayofweek * (2. * np.pi / 7))
    feature_df[str(feature_name) + 'day_sin'] = np.sin((date_time.dt.day - 1) * (2. * np.pi / 31))
    feature_df[str(feature_name) + 'day_cos'] = np.cos((date_time.dt.day - 1) * (2. * np.pi / 31))
    feature_df[str(feature_name) + 'mnth_sin'] = np.sin((date_time.dt.month - 1) * (2. * np.pi / 12))
    feature_df[str(feature_name) + 'mnth_cos'] = np.cos((date_time.dt.month - 1) * (2. * np.pi / 12))


def add_int_feature(original_df, feature_df, feature_name):
    feature_df[feature_name] = original_df[feature_name]

    if feature_name == "weight":
        # 2 kg --> 2.204 lbs
        feature_df[feature_name] = original_df[feature_name] * original_df["weight_units"].replace(2, 2.20462)


def add_binary_feature(original_df, feature_df, feature_name, one_value):
    feature_df[feature_name] = original_df[feature_name] == one_value


def add_datetime_feature(original_df, feature_df, feature_name):
    date_time = pd.to_datetime(original_df[feature_name].str.slice(0, 19), infer_datetime_format=True)
    feature_df[str(feature_name) + 'day_week_sin'] = np.sin(date_time.dt.dayofweek * (2. * np.pi / 7))
    feature_df[str(feature_name) + 'day_week_cos'] = np.cos(date_time.dt.dayofweek * (2. * np.pi / 7))
    feature_df[str(feature_name) + 'day_sin'] = np.sin((date_time.dt.day - 1) * (2. * np.pi / 31))
    feature_df[str(feature_name) + 'day_cos'] = np.cos((date_time.dt.day - 1) * (2. * np.pi / 31))
    feature_df[str(feature_name) + 'mnth_sin'] = np.sin((date_time.dt.month - 1) * (2. * np.pi / 12))
    feature_df[str(feature_name) + 'mnth_cos'] = np.cos((date_time.dt.month - 1) * (2. * np.pi / 12))
