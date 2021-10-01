import pandas as pd


def preprocess(address):
    original_df = load_dataset(address)
    feature_df = create_empty_df()
    add_binary_feature(original_df, feature_df, "b2c_c2c", "B2C")
    add_int_feature(original_df, feature_df, "declared_handling_days")
    add_int_feature(original_df, feature_df, "shipping_fee")
    add_int_feature(original_df, feature_df, "carrier_min_estimate")
    add_int_feature(original_df, feature_df, "carrier_max_estimate")
    add_int_feature(original_df, feature_df, "item_price")
    add_int_feature(original_df, feature_df, "quantity")
    add_int_feature(original_df, feature_df, "weight")
    add_categorical_feature(original_df, feature_df, "shipment_method_id")
    add_categorical_feature(original_df, feature_df, "category_id")
    add_categorical_feature(original_df, feature_df, "package_size")

    return feature_df


def load_dataset(address):
    return pd.read_csv(address, sep="\t")


def calculate_label(original_df):
    "payment_datetime"
    "delivery_date"


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
