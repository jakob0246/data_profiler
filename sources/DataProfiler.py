import time

import numpy as np
import pandas as pd
# import pandas_profiling as pp
from scipy.stats import normaltest
from sklearn.neighbors import LocalOutlierFactor


def determine_normal_distributions(dataframe):
    # drop categoricals: (because ordinal values arent scope of thesis? and nominal values cant be modelled after a distribution)
    columns_to_drop = dataframe.select_dtypes(include=['category']).columns
    dataframe = dataframe.drop(columns=columns_to_drop)

    # TODO: can be removed in the future
    dataframe = dataframe.dropna()

    alpha = 0.05  # 0.05

    results_array = np.zeros((dataframe.shape[1], 1))
    for i, column_name in enumerate(dataframe.columns):
        column = np.array(dataframe[column_name])

        statistics, p = normaltest(column)
        results_array[i] = p > alpha

    return results_array


def determine_outliers(dataframe):
    classifier = LocalOutlierFactor(algorithm="kd_tree")
    y_pred = classifier.fit_predict(dataframe)
    n_outliers = np.unique(y_pred, return_counts=True)[1][0]

    return n_outliers


def determine_class_std_deviation(dataframe, class_column):
    values_per_class = dataframe[class_column].value_counts()
    std_deviation = values_per_class.std()

    return std_deviation


def get_class_distribution(dataframe, class_column):
    return dataframe[class_column].value_counts()


def determine_n_missings_per_feature(dataframe_initial, class_column):
    if class_column != "" and class_column in dataframe_initial.columns:
        dataframe_initial = dataframe_initial.drop(columns=[class_column])

    n_missings_per_feature = {}
    for column in dataframe_initial.columns:
        n_missings_per_feature[column] = 0
        for value in dataframe_initial[column]:
            if not isinstance(value, str) and np.allclose(value, np.nan, equal_nan=True):
                n_missings_per_feature[column] += 1

    return n_missings_per_feature


def profile_data(dataframe_intitial, dataframe, supervised, class_column):
    # TODO: check if Unix or Windows for CPU time, because clock() does not return CPU time for Windows, it returns the normal time

    print("-*- [Data Profiler] Profiling the data ...")

    profiling_time_start = time.time()
    profiling_cputime_start = time.process_time()

    # describe for number-values
    pandas_data_profile_num = dataframe.describe()

    if len(dataframe.select_dtypes(include=['category']).columns) != 0:
        pandas_data_profile_cat = dataframe.describe(include=["category"])
    else:
        pandas_data_profile_cat = None

    # TODO: uncomment?
    # Profiled Data from Pandas-Profiling:
    # pandas_profiling_profile = pp.ProfileReport(dataframe)
    # pandas_profiling_dict = json.loads(pandas_profiling_profile.to_json())
    # pandas_profiling_dict_desc = pandas_profiling_profile.get_description()

    dataframe_missing_class = dataframe.copy()
    dataframe_initial_missing_class = dataframe_intitial.copy()
    if supervised:
        dataframe_missing_class = dataframe_missing_class.drop(columns=[class_column])
        dataframe_initial_missing_class = dataframe_initial_missing_class.drop(columns=[class_column])

    # get outliers:
    n_outliers = 0  # determine_outliers(dataframe_missing_class)

    # get singlevariate normal distributions:
    distributions = determine_normal_distributions(dataframe_initial_missing_class)

    # compute high pairwise correlation percentage:
    correlation_threshold = 0.66
    corr_matrix = np.array(dataframe_missing_class.corr())
    n_high_correlations = 0
    for i in range(0, corr_matrix.shape[0] - 1):
        for j in range(i + 1, corr_matrix.shape[0]):
            if corr_matrix[i, j] >= correlation_threshold or corr_matrix[i, j] <= (-1) * correlation_threshold:
                n_high_correlations += 1
    correlation_percentage = n_high_correlations / (0.5 * (corr_matrix.shape[0] ** 2 - corr_matrix.shape[0]))

    if supervised:
        class_std_deviation = determine_class_std_deviation(dataframe, class_column)
    else:
        class_std_deviation = None

    n_missing_values_per_column = pd.Series(((np.ones(dataframe_intitial.shape[1]) * dataframe_intitial.shape[0]) - dataframe_intitial.count().values).astype(int), index=dataframe_intitial.columns)
    n_missing_values_total = np.sum(n_missing_values_per_column)

    n_categorical_columns = len(dataframe_initial_missing_class.select_dtypes(include=['category']).columns)
    n_numerical_columns = dataframe_initial_missing_class.shape[1] - n_categorical_columns

    data_profile = {
        "dtypes": dataframe.dtypes,
        "n_rows": dataframe.shape[0],
        "n_columns": dataframe.shape[1],
        "n_values": dataframe.count(),
        "n_classes": len(dataframe[class_column].unique()) if supervised else 0,
        "n_numerical_columns": n_numerical_columns,
        "n_categorical_columns": n_categorical_columns,
        "n_missing_values_total": n_missing_values_total,
        "n_missing_values_per_column": n_missing_values_per_column,
        "correlation": dataframe.corr(),
        "covariance": dataframe.cov(),
        "min": pandas_data_profile_num.T["min"].copy(),
        "max": pandas_data_profile_num.T["max"].copy(),
        "mean": pandas_data_profile_num.T["mean"].copy(),
        "std_deviation": pandas_data_profile_num.T["std"].copy(),
        "25_percentile": pandas_data_profile_num.T["25%"].copy(),
        "50_percentile": pandas_data_profile_num.T["50%"].copy(),
        "75_percentile": pandas_data_profile_num.T["75%"].copy(),
        "unique": None if pandas_data_profile_cat is None else pandas_data_profile_cat.T["unique"].copy(),
        "top": None if pandas_data_profile_cat is None else pandas_data_profile_cat.T["unique"].copy(),
        "freq": None if pandas_data_profile_cat is None else pandas_data_profile_cat.T["unique"].copy(),
        "n_outliers": n_outliers,
        "outlier_percentage": n_outliers / dataframe_missing_class.shape[0],
        "n_normal_distributions": sum(distributions)[0] if len(distributions) != 0 else 0,
        "normal_distribution_percentage": (sum(distributions)[0] / dataframe_missing_class.shape[1]) if len(distributions) != 0 else 0,
        "high_correlation_percentage": correlation_percentage,
        "class_std_deviation": class_std_deviation,
        "class_distribution": get_class_distribution(dataframe, class_column) if supervised else pd.Series(),
        "n_missings_per_feature": determine_n_missings_per_feature(dataframe_intitial, class_column)
    }

    profiling_time_end = time.time()
    profiling_cputime_end = time.process_time()
    print(f"-*- [Data Profiler] Data profiled! Took {(profiling_time_end - profiling_time_start):.6f}s (Real time); {(profiling_cputime_end - profiling_cputime_start):.6f} s (CPU time)")

    return data_profile
