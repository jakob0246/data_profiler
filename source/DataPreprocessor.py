import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


class MissingValueHandlingTypes:
    cca = "cca"
    aca = "aca"
    impute = "impute"


def initial_preprocessing(dataframe, numeric_categorials, class_column, learning_type):
    # check dataset-specific constraints:
    if learning_type == "supervised" and class_column not in dataframe.columns:
        raise RuntimeError("the specified class column \"" + class_column + "\" from the config is not part of the dataset! " +
                           "(which is needed when supervised learning should be performed)")

    # replace missing values (if columns aren't all numeric dtypes -> would return error)
    if len(dataframe.select_dtypes([np.object]).columns) != 0:
        dataframe = dataframe.replace({b"?": np.nan})

    # convert binary-strings to strings:
    for column in dataframe.select_dtypes([np.object]).columns:
        dataframe[column] = dataframe[column].str.decode("utf-8")

    # convert objects to categoricals:
    for column in dataframe.select_dtypes([np.object]).columns:
        dataframe[column] = dataframe[column].astype("category")

    # convert numeric categoricals to floats:
    for column in numeric_categorials:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("float64")

    return dataframe


def handle_missing_values(dataset, config_parameter):
    dataset_modified = dataset.copy()

    if config_parameter == "cca":
        dataset_modified = dataset_modified.dropna()
    elif config_parameter == "aca":
        dataset_modified = dataset_modified.dropna(axis='columns')
    elif config_parameter == "impute":
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataset_modified = imputer.fit_transform(dataset_modified)
        dataset_modified = pd.DataFrame(dataset_modified, columns=dataset.columns)

    assert not dataset_modified.empty, "missing value handler dropped all values of the dataset! maybe try a different " \
                                       "handling method regarding missing values"

    return dataset_modified


def further_preprocessing(dataframe, missing_value_parameter):
    # handle missing values (CCA, ACA or imputation):
    dataframe = handle_missing_values(dataframe, missing_value_parameter)

    return dataframe


def prepare_for_unsupervised_learning(dataframe, numeric_categorials, class_column):
    # remove class-column:
    if class_column != "" and class_column in dataframe.columns:
        dataframe = dataframe.drop(columns=[class_column])

    # do one-hot-encoding (program just supports ordinal values!):
    columns_to_be_encoded = list(set(dataframe.select_dtypes(include=['category']).columns) - set(numeric_categorials))
    selection_to_be_encoded = dataframe[columns_to_be_encoded]

    encoder = OneHotEncoder()
    selection_transformed = encoder.fit_transform(selection_to_be_encoded).toarray()
    dataframe_selection_transformed = pd.DataFrame(selection_transformed, columns=list(map(lambda ele: "<ohe>_" + ele, encoder.get_feature_names(columns_to_be_encoded))))

    dataframe = dataframe.reset_index(drop=True)
    dataframe_selection_transformed = dataframe_selection_transformed.reset_index(drop=True)
    dataframe = dataframe.join(dataframe_selection_transformed)

    dataframe = dataframe.drop(columns=columns_to_be_encoded)

    return dataframe


def prepare_for_supervised_learning(dataframe, numeric_categorials, class_column):
    # do one-hot-encoding:
    columns_to_be_encoded = list(set(dataframe.select_dtypes(include=['category']).columns) - set(numeric_categorials) - {class_column})
    selection_to_be_encoded = dataframe[columns_to_be_encoded]

    encoder = OneHotEncoder()
    selection_transformed = encoder.fit_transform(selection_to_be_encoded).toarray()
    dataframe_selection_transformed = pd.DataFrame(selection_transformed, columns=list(map(lambda ele: "<ohe>_" + ele, encoder.get_feature_names(columns_to_be_encoded))))

    dataframe = dataframe.reset_index(drop=True)
    dataframe_selection_transformed = dataframe_selection_transformed.reset_index(drop=True)
    dataframe = dataframe.join(dataframe_selection_transformed)

    dataframe = dataframe.drop(columns=columns_to_be_encoded)

    return dataframe


def quantile_scale_and_normalize(dataset, feature):
    dataset_transformed = dataset.copy()

    n_quantiles = dataset.shape[0] // 10

    scaler = QuantileTransformer(n_quantiles=n_quantiles)

    feature_array = np.array(dataset_transformed[feature])
    feature_array_reshaped = np.reshape(feature_array, (len(feature_array), 1))

    feature_array_reshaped = scaler.fit_transform(feature_array_reshaped)
    feature_array_original_shape = np.reshape(feature_array_reshaped, (1, len(feature_array_reshaped)))[0]
    dataset_transformed[feature] = pd.Series(feature_array_original_shape)

    return dataset_transformed


def scale_and_normalize_features(dataset, class_column, supervised):
    dataset_transformed = dataset.copy()

    features_to_normalize = set(dataset.columns) - set([class_column]) if supervised else set(dataset.columns)
    for feature in features_to_normalize:
        dataset_transformed = quantile_scale_and_normalize(dataset_transformed, feature)

    return dataset_transformed