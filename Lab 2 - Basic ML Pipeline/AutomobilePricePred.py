import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load automobile price data
# ENSURE "AutomobilePrice.csv" is under your working directory
data = pd.read_csv("AutomobilePrice_Lab2.csv")
# data should contain 205 rows and 26 columns
print('\n\ndata.info():')
print(data.info())

# data preprocessing
# remove the 'normalized-losses' column
data.drop("normalized-losses", axis=1, inplace=True)

# remove rows that contain missing values
# first replace '?' in the data with special values 'nan'
# 'nan' (not a number) is defined in the numpy module
data.replace(to_replace='?', value=np.nan, inplace=True)
# remove any rows that contain 'nan'
data.dropna(axis=0, how="any", inplace=True)

# select a subset of features for regression:
# 'make', 'body-style', 'wheel-base', 'engine-size', 'horsepower',
# 'peak-rpm', 'highway-mpg', 'price'
data_subset = data.loc[:, ['make', 'body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg', 'price']]

# ENSURE 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm',
# 'highway-mpg', and 'price' are handled as 'float64' type
data_subset[['wheel-base', 'engine-size', 'horsepower', 'peak-rpm','highway-mpg', 'price']] \
= data_subset[['wheel-base', 'engine-size', 'horsepower', 'peak-rpm','highway-mpg', 'price']].astype(float)

# encode categorical values in 'make' and 'body-style' using one hot encoding
# which convert each category value into a new column and assigns a 1 or 0
# (True/False) value to the column
data_subset = pd.get_dummies(data_subset, columns = ['make', 'body-style'],
                             prefix=['make', 'body-style'], drop_first=True)

# data_subset now should contain 193 rows and 32 columns
print('\n\ndata_subset.info():')
print(data_subset.info())

# split the data into training set and test set
# use 75 percent of the data to train the model and hold back 25 percent
# for testing
train_ratio = 0.75
# number of samples in the data_subset
num_rows = data_subset.shape[0]
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
train_data = data_subset.iloc[train_indices, :]
test_data = data_subset.iloc[test_indices, :]
print(len(train_data), "training samples + ", len(test_data), "test samples")

# prepare training features and training labels
# features: all columns except 'price'
# labels: 'price' column
train_features = train_data.drop('price', axis=1, inplace=False)
train_labels = train_data.loc[:, 'price']

# prepare test features and test labels
test_features = test_data.drop('price', axis=1, inplace=False)
test_labels = test_data.loc[:, 'price']

# train a linear regression model using training data
lin_reg = LinearRegression()
# Note: you can ignore warning messages regarding gelsd driver
lin_reg.fit(train_features, train_labels)

# predict new prices on test data
price_pred = lin_reg.predict(test_features)

# compute mean absolute error
# formula: MAE = mean of | y_i - y_i_pred |
# where y_i is the i-th element of test_labels
#       y_i_pred is the i-th element of the price_pred
# Ref: https://en.wikipedia.org/wiki/Mean_absolute_error
y_true = test_labels.to_numpy()
y_pred = price_pred

mae = np.mean(np.abs(y_true - y_pred))
print('Mean Absolute Error = ', mae)

# compute root means square error
# formula: RMSE = square root of mean of ( y_i - y_i_pred )^2
# Ref: https://en.wikipedia.org/wiki/Root-mean-square_deviation
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
print('Root Mean Squared Error = ', rmse)

# compute coefficient of determination (aka R squared)
# formula: CoD = 1 - SSres/SStot, where
# SSres = sum of squares of ( y_i - y_i_pred )
# SStot = sum of squares of ( y_i - mean of y_i )
# Ref: https://en.wikipedia.org/wiki/Coefficient_of_determination
total_sum_sq = np.sum((y_true * y_true) **2)
res_sum_sq = np.sum((y_true * np.mean(y_true)) **2)
CoD = 1 - (res_sum_sq/total_sum_sq)
print('Coefficient of Determination = ', CoD)

# ------ Lab Part 2 ------

# Idea:
# Use the base training data and test for the mae value resulting
#     from using only single columns as the feature.
# Have a loop that attempts each value with each of the features and
# return the feature that resulted in the lowest mae

print("Lab 1 - Part 2")

per_feature_mae = {}

for col in list(train_features.columns):
    # Train and test on only a single column
    x_train_col = train_features[[col]].astype(float) # cast bool dummies to 0/1
    x_test_col = test_features[[col]].astype(float)

    # fit and predict with a fresh model
    model_col = LinearRegression()
    model_col.fit(x_train_col, train_labels)

    y_pred_col = model_col.predict(x_test_col)

    # Compute this feature's MAE
    mae_col = mean_absolute_error(test_labels, y_pred_col)
    per_feature_mae[col] = mae_col
    print(f'MAE using {col}: {mae_col:.4f}')

best_feature = min(per_feature_mae, key=per_feature_mae.get)
print(f'\nBest Single Feature: {best_feature}\n Feature MAE: {per_feature_mae[best_feature]:.4f}')