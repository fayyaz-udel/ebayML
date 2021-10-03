import logging

import pandas as pd


def preprocess(train_address, test_address, no_of_sample, save_to_file):
    logging.info("1")
    original_df, train_end_index = load_dataset(train_address, test_address, no_of_sample)
    logging.info("2")
    logging.info("train index: " + str(train_end_index))
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
    add_datetime_feature(original_df, feature_df, "acceptance_scan_timestamp")
    logging.info("15")
    label = calculate_label(original_df[:train_end_index], "acceptance_scan_timestamp")

    feature_df = feature_df.fillna(0)
    label = label.fillna(0)

    logging.info("15")
    if save_to_file:
        save_df(feature_df, 'x')
    logging.info("16")
    if save_to_file:
        save_df(label, 'y')

    return feature_df[:train_end_index], feature_df[train_end_index:], label


def save_df(df, name):
    df.to_csv(str(name) + ".csv")


def load_dataset(train_address, test_address, no_of_sample):
    train_df = pd.read_csv(train_address, sep="\t")

    train_df = train_df[:no_of_sample]
    train_df = clean_dataset(train_df)
    test_df = pd.read_csv(test_address, sep="\t")
    return pd.concat([train_df, test_df]), train_df.shape[0]


def clean_dataset(df):
    logging.info("30")
    df = df[df["shipping_fee"] >= 0]
    logging.info("31")
    df = df[df["carrier_min_estimate"] >= 0]
    logging.info("32")
    df = df[df["carrier_max_estimate"] >= 0]

    df = df[
        (pd.to_datetime(df["delivery_date"]) - pd.to_datetime(df["acceptance_scan_timestamp"].str.slice(0, 10))).dt.days >= 0]
    logging.info("34")
    df = df[
        (pd.to_datetime(df["acceptance_scan_timestamp"].str.slice(0, 10)) - pd.to_datetime(
            df["payment_datetime"].str.slice(0, 10))).dt.days >= 0]

    return df


def calculate_label(original_df, start_point):
    start = pd.to_datetime(original_df[start_point].str.slice(0, 10))
    logging.info("100")
    end = pd.to_datetime(original_df["delivery_date"])
    logging.info("101")
    delta = (end - start).dt.days
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


def add_datetime_feature(original_df, feature_df, feature_name):
    logging.info("30")
    date_time = pd.to_datetime(original_df[feature_name])
    logging.info("31")
    feature_df[str(feature_name) + "_hour_of_day"] = date_time.dt.hour
    logging.info("32")
    feature_df[str(feature_name) + "_day_of_week"] = date_time.dt.dayofweek
    logging.info("33")
    feature_df[str(feature_name) + "_day_of_month"] = date_time.dt.day
    logging.info("34")
    feature_df[str(feature_name) + "_month_of_year"] = date_time.dt.month

    return None
