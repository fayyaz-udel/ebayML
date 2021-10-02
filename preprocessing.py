import pandas as pd
import swifter
import logging

from pandas import Timestamp


def preprocess(address, save_to_file):
    logging.info("1")
    original_df = load_dataset(address)
    logging.info("2")
    feature_df = create_empty_df()
    logging.info("3")
    add_binary_feature(original_df, feature_df, "b2c_c2c", "B2C")
    logging.info("4")
    add_int_feature(original_df, feature_df, "declared_handling_days")
    logging.info("5")
    add_int_feature(original_df, feature_df, "shipping_fee")
    logging.info("6")
    add_int_feature(original_df, feature_df, "carrier_min_estimate")
    logging.info("7")
    add_int_feature(original_df, feature_df, "carrier_max_estimate")
    logging.info("8")
    add_int_feature(original_df, feature_df, "item_price")
    logging.info("9")
    add_int_feature(original_df, feature_df, "quantity")
    logging.info("10")
    add_int_feature(original_df, feature_df, "weight")
    logging.info("11")
    add_categorical_feature(original_df, feature_df, "shipment_method_id")
    logging.info("12")
    add_categorical_feature(original_df, feature_df, "category_id")
    logging.info("13")
    add_categorical_feature(original_df, feature_df, "package_size")
    logging.info("14")
    label = calculate_label(original_df)

    feature_df = feature_df.fillna(0)
    label = label.fillna(0)

    logging.info("15")
    if save_to_file:
        save_df(feature_df, 'x')
    logging.info("16")
    if save_to_file:
        save_df(label, 'y')

    return feature_df, label


def save_df(df, name):
    df.to_csv(str(name) + ".csv")


def load_dataset(address):
    return pd.read_csv(address, sep="\t")


def calculate_label(original_df):
    start = pd.to_datetime(original_df["payment_datetime"].str.slice(0,10))#.apply(lambda x: x.date())
    logging.info("100")
    end = pd.to_datetime(original_df["delivery_date"])#.apply(lambda x: x.date())
    logging.info("101")
    delta = (end - start).dt.days#.apply(lambda x: x.days)
    return delta


def create_empty_df():
    return pd.DataFrame()


def add_time_feature(original_df, feature_df, feature_name):
    return feature_df


def add_categorical_feature(original_df, feature_df, feature_name):
    dummmies = pd.get_dummies(original_df[feature_name], prefix=feature_name)
    for col in dummmies:
        feature_df[col] = dummmies[col]

    return feature_df


def add_int_feature(original_df, feature_df, feature_name):
    feature_df[feature_name] = original_df[feature_name]

    if feature_name == "weight":
        # 2 kg --> 2.204 lbs
        feature_df[feature_name] = original_df[feature_name] * original_df["weight_units"].replace(2, 2.20462)

    return feature_df


def add_binary_feature(original_df, feature_df, feature_name, one_value):
    feature_df[feature_name] = original_df[feature_name] == one_value
