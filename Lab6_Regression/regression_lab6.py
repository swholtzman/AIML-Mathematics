from typing import Any, Callable

import loaddata_lab6 as loader
import pandas as pd
import numpy as np
import sklearn.linear_model as sk
import matplotlib.pyplot as plt
import warnings

from numpy import floating, ndarray, dtype, float64
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.base import clone


class BaseRegression:
    """
    Shared logic for data splitting and evaluation.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.training_data, self.test_data = self.custom_split(data, 0.3)
        self.beta = None
        self.model = None
        self.feature_columns = None
        self.scaler = None

    @staticmethod
    def custom_split(data: pd.DataFrame, train_frac: float, random_state: int = 42) -> tuple:
        """
        Manual implementation of sklearn.model_selection.train_test_split

        :param data: The full dataset to split.
        :param train_frac: The fraction of data to be used for training (e.g., 0.3).
        :param random_state: Seed for the random number generator for reproducibility.
        :return: A tuple containing (train_dataframe, test_dataframe).
        """
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(len(data))
        train_size = int(len(data) * train_frac)

        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]

        train_dataframe = data.iloc[train_indices]
        test_dataframe = data.iloc[test_indices]

        return train_dataframe, test_dataframe

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        Calculates the Mean Absolute Error (MAE).

        :param y_true: True values
        :param y_pred: Predicted values
        :return: Mean Absolute Error; A measurement of prediction error.
        """

        return np.mean(np.abs(y_true - y_pred))

    def cross_validation_split(self,
                               model_prototype: Any,
                               data_x: pd.DataFrame,
                               data_y: pd.Series,
                               k_folds: int,
                               random_state: int = 42) -> tuple:
        """
        Performs K-fold cross-validation on a given model prototype.

        :param model_prototype: An sklearn-compatible model instance
                                (e.g., sk.Ridge(alpha=1.0)). This model
                                will be cloned (re-initialized) for each fold.
        :param data_x: The full training feature set (pd.DataFrame).
        :param data_y: The full training target set (pd.Series).
        :param k_folds: The number of folds (e.g., 5).
        :param random_state: Random seed for shuffling.
        :return:
            avg_train_error:    Average MAE across all K training folds.
            avg_cv_error:       Average MAE across all K validation (hold-out) folds.
        """

        # Step 1: KFold splitter
        kfold_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        # Ensure data is in numpy format for splitting and fitting
        x_np = np.asarray(data_x)
        y_np = np.asarray(data_y)

        # Step 2: Create lists to store the error of each fold
        train_errors_kfold = []
        cross_validation_errors_kfold = []

        # Step 3: Loop through each fold
        for fold_index, (train_indices, val_indices) in enumerate(kfold_splitter.split(x_np)):
            # Separate data into training and validation sets within this fold
            x_train_fold, y_train_fold = x_np[train_indices], y_np[train_indices]
            x_val_fold, y_val_fold = x_np[val_indices], y_np[val_indices]

            # --- Model Fitting ---
            # Clone the protocol model.
            # Necessary for getting a fresh & unfitted model per fold
            fold_model = clone(model_prototype)

            # Train the model on this fold's training data
            fold_model.fit(x_train_fold, y_train_fold)

            # Make predictions on both this fold's training and validation sets
            y_train_prediction = fold_model.predict(x_train_fold)
            y_val_prediction = fold_model.predict(x_val_fold)

            # Calculate MAE for this fold
            fold_train_error = self.mean_absolute_error(y_train_fold, y_train_prediction)
            fold_cross_val_error = self.mean_absolute_error(y_val_fold, y_val_prediction)

            # Add errors to the lists
            train_errors_kfold.append(fold_train_error)
            cross_validation_errors_kfold.append(fold_cross_val_error)

        # 4. Calculate the average error across all K folds
        avg_train_error = np.mean(train_errors_kfold)
        avg_cv_error = np.mean(cross_validation_errors_kfold)

        return float(avg_train_error), float(avg_cv_error)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using fitted sklearn Lasso model.
        (This function is identical to the Ridge one)

        :param X: Feature DataFrame (standardized).
        :return: Array of predictions.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        X_np = np.asarray(X, dtype=np.float64)

        # Do NOT add intercept column; sklearn's predict handles it
        return self.model.predict(X_np)

    def standardize_data(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Standardize all columns EXCEPT TARGET_D to zero mean and unit variance.
        This fits the scaler and transforms the data.

        :param dataframe: The full, *imputed* DataFrame.
        :return: A new DataFrame with features standardized.
        """
        y_col = "TARGET_D"

        # Get all column names except target column
        X_cols = [col for col in dataframe.columns if col != y_col]

        # --- Pre-scaling cleanup ---
        # Find constant columns (std_dev = 0) among features,
        #   as they cause division by zero in StandardScaler
        features_std = dataframe[X_cols].std()
        constant_feature_cols = features_std[features_std == 0].index

        if not constant_feature_cols.empty:
            print(f"Warning: Removing {len(constant_feature_cols)} constant feature columns "
                  f"to prevent division by zero.")
            # Drop from main list of feature columns
            X_cols = [col for col in X_cols if col not in constant_feature_cols]
            # And drop them from dataframe copy
            dataframe = dataframe.drop(columns=constant_feature_cols)
        # --- End pre-scaling cleanup ---

        # Store clean feature columns
        self.feature_columns = X_cols

        self.scaler = StandardScaler()

        # Fit scaler ONLY on non-constant feature columns
        X_standard = self.scaler.fit_transform(dataframe[X_cols]).astype(np.float64)

        # Create copy to hold standardized data
        dataframe_standard = dataframe.copy()

        # Overwrite feature columns with standardized versions
        dataframe_standard[X_cols] = X_standard

        print("Data standardization complete.")
        return dataframe_standard


class LinearRegressionModel(BaseRegression):
    """
    Implements OLS regression using the closed-form solution.
    """
    def fit(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Fit linear regression model using pseudo-inverse for stability
        Adds an intercept (column of ones) manually

        :param X: Feature DataFrame.
        :param y: Target Series
        :return: Beta coefficients (including intercept).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Add intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.beta = np.linalg.pinv(X) @ y
        return self.beta

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using learned coefficients.
        Manually adds an intercept term to X for prediction

        :param X: Feature DataFrame.
        :return: An array of predictions.
        """
        X_np = np.asarray(X, dtype=np.float64)

        X_with_intercept = np.hstack([np.ones((X_np.shape[0], 1)), X_np])
        return X_with_intercept @ self.beta

    def run_model(self) -> floating[Any]:
        """
        Fit and evaluate the OLS model.
        Implements Lab Part 1.

        :return: The Mean Absolute Error on the test set.
        """
        print("\n=== LAB 6 PART 1 - LINEAR REGRESSION ===")

        # Target and feature selection
        y_col = "TARGET_D"
        X_cols = [col for col in self.training_data.columns if col != y_col]

        X_train_full = self.training_data[X_cols]
        y_train_full = self.training_data[y_col]
        X_test_full = self.test_data[X_cols]
        y_test_full = self.test_data[y_col]

        # --- Constant Column Removal ---
        # Find constant columns in the training data
        features_std = X_train_full.std()
        constant_feature_cols = features_std[features_std == 0].index

        if not constant_feature_cols.empty:
            print(f"Part 1: Removing {len(constant_feature_cols)} constant feature columns.")
            # Drop them from the training and test sets
            X_train = X_train_full.drop(columns=constant_feature_cols)
            X_test = X_test_full.drop(columns=constant_feature_cols)
        else:
            X_train = X_train_full
            X_test = X_test_full
        # --- End Removal ---

        # Fit model
        self.fit(X_train, y_train_full)

        # Evaluate
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        mae_train = self.mean_absolute_error(y_train_full, y_pred_train)
        mae_test = self.mean_absolute_error(y_test_full, y_pred_test)

        print(f"Mean Absolute Error (MAE) for Training: {mae_train:.4f}")
        print(f"Mean Absolute Error (MAE) for Testing:  {mae_test:.4f}")
        print("=========================================")

        return mae_test


class RidgeRegressionModel(BaseRegression):
    def __init__(self, data: pd.DataFrame):
        # super() splits unstandardized data
        #   must split again after standardization
        super().__init__(data)
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] | None = None
        self.model: sk.Ridge | None = None

        # Standardize entire dataset first
        std_data = self.standardize_data(data)
        self.data = std_data

        # Split into train/test after standardizing
        self.training_data, self.test_data = self.custom_split(std_data, 0.3)
        self.beta = None

    def fit(self, X: pd.DataFrame, y: pd.Series, lam: float = 1.0) -> None:
        """
        Fit a Ridge regression model using sklearn.
        sklearn's Ridge handles the intercept automatically
            (fit_intercept=True by default), so do NOT add
            a column of ones.

        :param X: Feature DataFrame (standardized).
        :param y: Target Series.
        :param lam: Regularization strength (known as 'alpha' in sklearn).
        """
        self.model = sk.Ridge(alpha=lam, fit_intercept=True)

        # Data to numpy array for sklearn
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)

        self.model.fit(X_np, y_np)

        # Store coefficients for viewing
        self.beta = self.model.coef_

    def run_ridge_analysis(self, lambda_values: list | np.ndarray, k_folds: int = 5) -> float:
        """
        Performs Steps 4 & 5 of the lab:
        - Loops through all provided lambda values.
        - Calls `cross_validation_split` for each lambda to get avg train/CV MAE.
        - Plots the average MAE values vs. lambda (log scale).
        - Determines and returns the best lambda (which minimizes CV MAE).

        :param lambda_values: A list or array of lambda values to test.
        :param k_folds: The number of folds for cross-validation (e.g., 5).
        :return: The best lambda value (float).
        """
        print("\n=== LAB 6 PART 2 - RIDGE REGRESSION ===")

        # Get the full training dataset (pre-standardized via init)
        y_col = "TARGET_D"

        # Use stored feature columns
        X_cols = self.feature_columns

        X_train_full = self.training_data[X_cols]
        y_train_full = self.training_data[y_col]

        # Lists to store avg MAE for each Lambda
        avg_train_maes = []
        avg_cv_maes = []

        print(f"Starting {k_folds}-fold cross-validation for {len(lambda_values)} lambda values...")

        # --- Step 4 (Loop and Evaluate) ---
        for lam_val in lambda_values:
            # Create "prototype" model for this lambda
            model_prototype = sk.Ridge(alpha=lam_val, fit_intercept=True)

            # Use generic cross_validation_split from BaseRegresssion class
            train_mae, cross_val_mae = self.cross_validation_split(
                model_prototype = model_prototype,
                data_x=X_train_full,
                data_y=y_train_full,
                k_folds = k_folds,
            )

            # Store results
            avg_train_maes.append(train_mae)
            avg_cv_maes.append(cross_val_mae)

        print("Cross validation complete !")

        # --- Step 4: Plotting ---
        plt.figure(figsize=(10,6))
        plt.plot(lambda_values, avg_train_maes, 'bo-', label="Average Training MAE")
        plt.plot(lambda_values, avg_cv_maes, 'ro-', label="Average Cross-Validation MAE")

        # Use log scaling for x-axis (labda)
        plt.xscale('log')
        plt.xlabel('Lambda (Regularization Strength) - Log Scale')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Ridge Regression: MAE vs Lambda (5-fold CV)')
        plt.legend()
        plt.grid(True, which="both", ls="--")

        # Save plot to a file
        plot_filename = "ridge_mae_vs_lambda.png"
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")

        # --- Step 5: Determine Best Lambda) ---
        # Find the index of the minimum Cross-Validation MAE
        best_lambda_index = np.argmin(avg_cv_maes)
        best_lambda = lambda_values[best_lambda_index]

        print(f"\n--- Best Lambda Determination (Step 5) ---")
        print(f"Best Lambda (Minimized CV MAE): {best_lambda:e}")
        print(f" -> Minimum CV MAE: {avg_cv_maes[best_lambda_index]:.4f}")
        print(f" -> Training MAE at best lambda: {avg_train_maes[best_lambda_index]:.4f}")

        return float(best_lambda)


class LassoRegressionModel(BaseRegression):
    def __init__(self, data: pd.DataFrame):
        # super() splits unstandardized data
        #   must split again after standardization
        super().__init__(data)
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] | None = None
        self.model: sk.Lasso | None = None

        # Standardize the entire dataset first
        std_data = self.standardize_data(data)
        self.data = std_data

        # Split into train/test after standardizing
        self.training_data, self.test_data = self.custom_split(std_data, 0.3)
        self.beta = None

    def fit(self, X: pd.DataFrame, y: pd.Series, lam: float = 1.0) -> None:
        """
        Fit a Lasso regression model using sklearn.
        sklearn's Lasso handles intercept automatically
            (fit_intercept=True by default), so do NOT add
            a column of ones.

        :param X: Feature DataFrame (standardized).
        :param y: Target Series.
        :param lam: Regularization strength (known as 'alpha' in sklearn).
        """

        # --- Use sk.Lasso and add max_iter ---
        self.model = sk.Lasso(alpha=lam, fit_intercept=True, max_iter=100000)
        # --- max_iter is necessary for Lasso as algorithm is much more computationally
        #   heavy than the Ridge Regression algorithm ---

        # Data to numpy array for sklearn
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)

        self.model.fit(X_np, y_np)

        # Store coefficients for viewing
        self.beta = self.model.coef_

    def run_lasso_analysis(self, lambda_values: list | np.ndarray, k_folds: int = 5) -> float:
        """
        Performs analysis for Lasso regression:
        - Loops through all provided lambda values.
        - Calls `cross_validation_split` for each lambda to get avg train/CV MAE.
        - Plots the average MAE values vs. lambda (log scale).
        - Determines and returns the best lambda (which minimizes CV MAE)

        :param lambda_values: A list or array of lambda values to test.
        :param k_folds: The number of folds for cross-validation (e.g., 5).
        :return: The best lambda value (float)
        """
        print("\n=== LAB 6 PART 3 - LASSO REGRESSION CV ===")

        # Get full training dataset (pre-standardized via init)
        y_col = "TARGET_D"

        # Use stored feature columns
        X_cols = self.feature_columns

        X_train_full = self.training_data[X_cols]
        y_train_full = self.training_data[y_col]

        # Lists to store avg MAE for each Lambda
        avg_train_maes = []
        avg_cv_maes = []

        print(f"Starting {k_folds}-fold cross-validation for {len(lambda_values)} lambda values...")

        # --- Loop and Evaluate ---
        for lam_val in lambda_values:
            # Create "prototype" model for this lambda
            model_prototype = sk.Lasso(alpha=lam_val, fit_intercept=True, max_iter=100000)

            # Use generic cross_validation_split from BaseRegresssion classs
            train_mae, cross_val_mae = self.cross_validation_split(
                model_prototype=model_prototype,
                data_x=X_train_full,
                data_y=y_train_full,
                k_folds=k_folds,
            )

            # Store results
            avg_train_maes.append(train_mae)
            avg_cv_maes.append(cross_val_mae)

        print("Cross validation complete !")

        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        plt.plot(lambda_values, avg_train_maes, 'bo-', label="Average Training MAE")
        plt.plot(lambda_values, avg_cv_maes, 'ro-', label="Average Cross-Validation MAE")

        # Use log scaling for x-axis (labda)
        plt.xscale('log')
        plt.xlabel('Lambda (Regularization Strength) - Log Scale')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Lasso Regression: MAE vs Lambda (5-fold CV)')
        plt.legend()
        plt.grid(True, which="both", ls="--")

        # Save plot to a file
        plot_filename = "lasso_mae_vs_lambda.png"
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")

        # --- Determine Best Lambda ---
        # Find index of minimum Cross-Validation MAE
        best_lambda_index = np.argmin(avg_cv_maes)
        best_lambda = lambda_values[best_lambda_index]

        print(f"\n--- Best Lambda Determination (Lasso) ---")
        print(f"Best Lambda (Minimized CV MAE): {best_lambda:e}")
        print(f" -> Minimum CV MAE: {avg_cv_maes[best_lambda_index]:.4f}")
        print(f" -> Training MAE at best lambda: {avg_train_maes[best_lambda_index]:.4f}")

        return float(best_lambda)


def main():
    # Load data (Step 1)
    data = loader.LoadData().load("./data/data_lab6.csv")

    # --- DATA CLEANING ---
    # Replace infinite values with 0
    data = data.replace([np.inf, -np.inf], 0)

    # Impute all missing (NaN) values with 0
    data = data.fillna(0)
    print("Data cleaning (NaN/inf imputation) complete.")
    # --- END CLEANING ---

    # --- Part 1 (Linear Regression) ---
    # Handles splitting (Step 3) internally
    model = LinearRegressionModel(data)
    lin_reg_test_mae = model.run_model()

    # --- SUPPRESS WARNINGS HERE (NEW) ---
    print("\nSuppressing numerical RuntimeWarnings for Ridge Regression part...")
    # hide 'divide by zero', 'overflow', etc.
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    # --- END SUPPRESSION ---

    # ======================================
    # KEY LEARNING NOTES:
    # Linear regression fails with this model as there are too many parameters for too few samples.
    # The pre-processed data has over 3,000 features (p)
    # With only 30% of the ~8,900 rows for training (only ~2,700) samples (n)
    # As stated in text and in class, with p>>N, we fall into a "curse of dimensionality" problem
    # Attempting to solve OLS (stnd Linear Reg.), the XtX matrix determinant is 0 and cannot be inverted
    # Step 1's failures are shown in the console for learning, but are now unhelpful in step 2 and are therefore suppressed
    # ======================================

    # --- Part 2 (Ridge Regression) ---
    # Handles standardization (Step 2) and splitting (Step 3)
    ridge_model = RidgeRegressionModel(data)

    # Define lambda values (Step 4)
    # 14 values from 10^-3 to 10^10
    lambda_values = np.logspace(-3, 10, 14)

    # Run CV to find best lambda + plot (Steps 4 & 5)
    best_lambda = ridge_model.run_ridge_analysis(lambda_values, k_folds=5)

    # --- Step 6: Evaluate on Test Set ---
    print(f"\n--- Final Evaluation (Step 6) ---")
    print(f"Evaluating Ridge model with best lambda ({best_lambda:e}) on the test set...")

    # Get standardized training + test sets
    X_train_final = ridge_model.training_data[ridge_model.feature_columns]
    y_train_final = ridge_model.training_data["TARGET_D"]
    X_test_final = ridge_model.test_data[ridge_model.feature_columns]
    y_test_final = ridge_model.test_data["TARGET_D"]

    # Fit model on entire training set using best lambda
    ridge_model.fit(X_train_final, y_train_final, lam=best_lambda)

    # Predict on test set
    y_test_pred = ridge_model.predict(X_test_final)

    # Calculate + output final test MAE
    ridge_test_mae = ridge_model.mean_absolute_error(y_test_final, y_test_pred)
    print(f"Mean Absolute Error (MAE) for Testing (Ridge):  {ridge_test_mae:.4f}")
    print("===============================================")

    improvement = lin_reg_test_mae - ridge_test_mae

    # Check for meaningful improvement
    if improvement > 0.0001:
        percent_reduction = (improvement / lin_reg_test_mae) * 100
        print(f"Ridge regression improved the MAE by {improvement:.4f} (a {percent_reduction:.2f}% reduction).")
    elif improvement < -0.0001:
        print(f"Ridge regression performed worse, increasing MAE by {abs(improvement):.4f}.")
    else:
        print("Ridge regression performed almost identically to linear regression.")
    print("=================================")

    # --- Part 3 (Lasso Regression) ---
    print("\n--- Starting Part 3: Lasso ---")
    lasso_model = LassoRegressionModel(data)

    # 10^-2 to 10^2, with the exponent increasing by 0.25
    lasso_powers = np.arange(-2, 2.25, 0.25)
    lambda_values_lasso = 10 ** lasso_powers

    best_lambda_lasso = lasso_model.run_lasso_analysis(lambda_values_lasso, k_folds=5)

    # --- Final Lasso Evaluation ---
    print(f"\n--- Final Evaluation (Lasso) ---")
    X_train_lasso = lasso_model.training_data[lasso_model.feature_columns]
    y_train_lasso = lasso_model.training_data["TARGET_D"]
    X_test_lasso = lasso_model.test_data[lasso_model.feature_columns]
    y_test_lasso = lasso_model.test_data["TARGET_D"]

    lasso_model.fit(X_train_lasso, y_train_lasso, lam=best_lambda_lasso)
    y_test_pred_lasso = lasso_model.predict(X_test_lasso)

    lasso_test_mae = lasso_model.mean_absolute_error(y_test_lasso, y_test_pred_lasso)
    print(f"Mean Absolute Error (MAE) for Testing (Lasso):  {lasso_test_mae:.4f}")

    # --- Part 3, Step 6: Determine Top 3 Features ---
    print("\n--- Part 3, Step 6: Top 3 Lasso Features ---")

    # Get coefficients (betas) from fitted model
    lasso_betas = lasso_model.model.coef_

    # Get the corresponding feature names
    feature_names = lasso_model.feature_columns

    # Create a Pandas Series to map names to betas
    coef_series = pd.Series(lasso_betas, index=feature_names)

    # Get top 3 features by abs beta values
    top_3_features = coef_series.abs().nlargest(3)

    print("Top 3 features with the strongest effect (largest absolute coefficients):")
    for feature_name, abs_coef in top_3_features.items():
        # Get original beta (with its sign)
        original_coef = coef_series[feature_name]
        print(f"  - Feature: {feature_name}")
        print(f"    Coefficient: {original_coef:.4f} (Absolute Value: {abs_coef:.4f})")

    print("============================================")

    # --- Step 7: Compare Models ---
    print("\n=== LAB 6 COMPARISON (Step 7) ===")
    print(f"Linear Regression Test MAE: {lin_reg_test_mae:.4f}")
    print(f"Ridge Regression Test MAE:  {ridge_test_mae:.4f}")
    print(f"Lasso Regression Test MAE:  {lasso_test_mae:.4f}")
    print("=================================")

    # --- Part 3, Step 5: Determine Most Suitable Model ---
    print("\n--- Part 3, Step 5: Most Suitable Model ---")

    # Create dict of final test MAE values
    models_mae = {
        "Linear Regression": lin_reg_test_mae,
        "Ridge Regression": ridge_test_mae,
        "Lasso Regression": lasso_test_mae
    }

    # Find model name with minimum MAE (value)
    best_model_name = min(models_mae, key=models_mae.get)
    best_mae = models_mae[best_model_name]

    print(f"The most suitable model to predict donation amount is {best_model_name}")
    print(f"It achieved the lowest Test MAE: {best_mae:.4f}")
    print("==============================================")

    # ======
    # RIDGE REGRESSION DESCRIPTION (PART 2: STEP 7)
    # We can see the output is drastically different between part 1 and part 2
    # The MAE of part 1 (~27) is massive, while part 2 MAE is 8 (significant reduction in error)
    # The number of features vs samples (as described in comments above) pulls us into a problem of dimensionality
    #   where standard OLS methods cannot solve for the number of parameters the model requires.
    # Only by squaring all features (except the intercept) are we able to overcome this problem and find a reasonable MAE value
    # """
    # When sample sizes are small, reduce variance by reducing prediction's sensitivity to training data
    # 	- reduce variance
    # """ - https://www.youtube.com/watch?v=Q81RR3yKn30
    # Since we are able to reduce the sensitifity and improve our bias-variance measure on such a small data set (compared to size of feature set
    #   we can more accurately adjust our models measurements

    # LASSO REGRESSION ADDITION (PART 3: STEP 4)
    # The output of lasso is drastically better than the Linear Regression model, showing a similar effect on the MAE as
    # Ridge Regression was able to find, however it is slightly worse than the MAE found by Ridge Regression.
    # Lasso and Ridge are very similar models, but Lasso outperforms Ridge best when there are a lot of irrelevant features.
    # Finding that Ridge performed better on the unseen data than Lasso, indicates that the features that were used in this data set
    # are mostly relevant, and that the extra computation to reduce the influence of any unnecessary features to 0, was not worth it,
    # and brought down the overall performance of the model by negatively impacting those features that are important to the true function of this particular model.
    #


if __name__ == "__main__":
    main()

