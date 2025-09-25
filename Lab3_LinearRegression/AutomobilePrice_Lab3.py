import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_absolute_error

def comp4983_lin_reg_fit(X, y):
    """
    Fit linear regression with closed-form solution:
        w = (X^T X)^(-1) X^T y
    Adds an intercept column of 1s to X.
    """

    # === Use float64 ndarrays ===
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Add intercept (bias) column
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # (n_train, p+1)

    Xt = X.T
    XtX = np.dot(Xt, X)
    Xty = np.dot(Xt, y)

    XtXinv = np.linalg.inv(XtX)
    beta = np.dot(XtXinv, Xty)
    # print("XtXinvXty = \n", beta)

    return beta

def comp4983_lin_reg_predict(X, w):
    """
    Predict using learned weights w (intercept is w[0]).
        predict = X * \hat{beta}
        X = X Values
        w = beta
    """
    X = np.asarray(X, dtype=np.float64)

    # Add intercept (bias) column
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # (n_train, p+1)

    predict = np.dot(X, w)
    # print("Predict = \n", predict)

    return predict



data = pd.read_csv("AutomobilePrice_Lab3.csv")
train_ratio = 0.75

# number of samples in the data_subset
num_rows = data.shape[0]
# shuffle the indices
shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

# calculate the number of rows for training
train_set_size = int(train_ratio * num_rows)

# training set: take the first 'train_set_size' rows
train_indices = shuffled_indices[:train_set_size]
# test set: take the remaining rows
test_indices = shuffled_indices[train_set_size:]

# create training set and test set
train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]

train_features = train_data.drop('price', axis=1, inplace=False)
train_labels = train_data.loc[:, 'price']

# prepare test features and test labels
test_features = test_data.drop('price', axis=1, inplace=False)
test_labels = test_data.loc[:, 'price']



# ---+--- Lab 3 - Part 3 ---+---
beta = comp4983_lin_reg_fit(train_features, train_labels)
price_pred = comp4983_lin_reg_predict(test_features, beta)

y_true = test_labels.to_numpy()
y_pred = price_pred

# Ensure arrays are 1-D for consistency
actual_values   = np.asarray(y_true, dtype=float).ravel()
predicted_values = np.asarray(y_pred, dtype=float).ravel()

# --- Error metrics ---
mean_absolute_error_value = np.mean(np.abs(actual_values - predicted_values))
root_mean_squared_error_value = np.sqrt(np.mean((actual_values - predicted_values) ** 2))

# --- Components for R² ---
# Residual Sum of Squares (unexplained variation)
residual_sum_of_squares = np.sum((actual_values - predicted_values) ** 2)

# Total Sum of Squares (total variation around the mean)
total_sum_of_squares = np.sum((actual_values - np.mean(actual_values)) ** 2)

# Coefficient of Determination (R²)
coefficient_of_determination = 1.0 - (residual_sum_of_squares / total_sum_of_squares)

# --- Print results ---
print("Mean Absolute Error = ", mean_absolute_error_value)
print("Root Mean Squared Error = ", root_mean_squared_error_value)
print("Coefficient of Determination = ", coefficient_of_determination)




print("\n\n---+--- Lab 3 - Part 4 ---+---\n")

def fit_and_predict_with_columns(train_features, train_labels, test_features, column_names):
    """
    :param train_features: pandas.DataFrame of training features
    :param train_labels: pandas.Series of training labels (price)
    :param test_features: pandas.DataFrame of test features
    :param column_names: list[str] column names to use as predictors
    :return: np.ndarray predicted_values shaped (n_test,)
    """
    X_train_subset = train_features[column_names].astype(float).to_numpy()
    X_test_subset  = test_features[column_names].astype(float).to_numpy()
    y_train_values = train_labels.to_numpy()

    beta_values = comp4983_lin_reg_fit(X_train_subset, y_train_values)
    predicted_values = comp4983_lin_reg_predict(X_test_subset, beta_values)
    return np.asarray(predicted_values, dtype=float).ravel()

def compute_mae_rmse(actual_values, predicted_values):
    """
    :param actual_values: np.ndarray of true prices
    :param predicted_values: np.ndarray of predicted prices
    :return: (mean_absolute_error_value, root_mean_squared_error_value)
    """
    actual_values = np.asarray(actual_values, dtype=float).ravel()
    predicted_values = np.asarray(predicted_values, dtype=float).ravel()
    mean_absolute_error_value = mean_absolute_error(actual_values, predicted_values)
    root_mean_squared_error_value = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    return mean_absolute_error_value, root_mean_squared_error_value


def feature_pairs_generator(feature_list):
    feature_pair_set = []
    for feature1 in feature_list:
        for feature2 in feature_list:
            if feature1 == feature2:
                continue
            feature_pair_set.append((feature1, feature2))
    for feature in feature_pair_set:
        yield feature

def evaluate_single_feature_models(train_features, train_labels, test_features, test_labels):
    """
    :return: dict[str, dict[str, float]] keyed by feature name with MAE/RMSE
    """
    metrics_by_feature = {}
    for feature_name in list(train_features.columns):
        predicted_values = fit_and_predict_with_columns(
            train_features, train_labels, test_features, [feature_name]
        )
        mae_value, rmse_value = compute_mae_rmse(test_labels.to_numpy(), predicted_values)
        metrics_by_feature[feature_name] = {
            "mean_absolute_error": mae_value,
            "root_mean_squared_error": rmse_value,
        }
        # print(f'Metrics using single feature "{feature_name}": '
        #       f'MAE={mae_value:.4f}, RMSE={rmse_value:.4f}')
    return metrics_by_feature

def evaluate_feature_pair_models(train_features, train_labels, test_features, test_labels):
    """
    Uses your permutation-style pairing (A,B) and (B,A) both included.
    :return: dict[tuple[str,str], dict[str, float]] keyed by (feature1, feature2)
    """
    metrics_by_pair = {}
    for feature1, feature2 in feature_pairs_generator(list(train_features.columns)):
        predicted_values = fit_and_predict_with_columns(
            train_features, train_labels, test_features, [feature1, feature2]
        )
        mae_value, rmse_value = compute_mae_rmse(test_labels.to_numpy(), predicted_values)
        metrics_by_pair[(feature1, feature2)] = {
            "mean_absolute_error": mae_value,
            "root_mean_squared_error": rmse_value,
        }
        # print(f'Metrics using feature pair ("{feature1}", "{feature2}"): '
        #       f'MAE={mae_value:.4f}, RMSE={rmse_value:.4f}')
    return metrics_by_pair

def report_best_results(metrics_by_feature, metrics_by_pair):
    # best single feature (by MAE)
    best_single_feature = min(metrics_by_feature, key=lambda k: metrics_by_feature[k]["mean_absolute_error"])
    best_single_metrics = metrics_by_feature[best_single_feature]

    # best pair (by MAE)
    best_feature_pair = min(metrics_by_pair, key=lambda k: metrics_by_pair[k]["mean_absolute_error"])
    best_pair_metrics = metrics_by_pair[best_feature_pair]

    print("--- Best Single Feature (by MAE) ---")
    print(f'Feature: {best_single_feature}')
    print(f'MAE:  {best_single_metrics["mean_absolute_error"]:.4f}')
    print(f'RMSE: {best_single_metrics["root_mean_squared_error"]:.4f}')

    print("\n--- Best Feature Pair (by MAE) ---")
    print(f'Pair: {best_feature_pair}')
    print(f'MAE:  {best_pair_metrics["mean_absolute_error"]:.4f}')
    print(f'RMSE: {best_pair_metrics["root_mean_squared_error"]:.4f}')


single_feature_results = evaluate_single_feature_models(
    train_features, train_labels, test_features, test_labels
)

pair_results = evaluate_feature_pair_models(
    train_features, train_labels, test_features, test_labels
)

report_best_results(single_feature_results, pair_results)