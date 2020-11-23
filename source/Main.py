from source.DataIntegrator import read_in_data
from source.DataPreprocessor import initial_preprocessing, further_preprocessing, prepare_for_unsupervised_learning, \
                             prepare_for_supervised_learning, scale_and_normalize_features, MissingValueHandlingTypes
from source.DataProfiler import profile_data


supervised = False
missing_value_handling = MissingValueHandlingTypes.cca  # cca, aca, impute

path = "/datasets/agaricus-lepiota.csv"
delimiter = " "
class_column = "class"
numeric_categoricals = []

dataset_raw = read_in_data(path, delimiter)

dataset_initial_preprocessed = initial_preprocessing(dataset_raw, numeric_categoricals, class_column, supervised)
dataset_preprocessed = further_preprocessing(dataset_initial_preprocessed, missing_value_handling)

# prepare the dataset based on the learning type
if not supervised:
    dataset_preprocessed = prepare_for_unsupervised_learning(dataset_preprocessed, numeric_categoricals, class_column)
else:
    dataset_preprocessed = prepare_for_supervised_learning(dataset_preprocessed, numeric_categoricals, class_column)

# scale and normalize features
dataset_preprocessed = scale_and_normalize_features(dataset_preprocessed, class_column, supervised)

data_profile = profile_data(dataset_initial_preprocessed, dataset_preprocessed, supervised, class_column)
print(data_profile)
