import numpy as np
import pandas as pd


class CrossValidationLab5:
    def __init__(self):
        # Store the most recent results for standardized printing (__str__)
        self._last_p = None
        self._last_train_error = None
        self._last_cv_error = None

        self._highest_cv_error = None
        self._highest_p = None
        self._highest_train_error = None

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        :param y_true: True values
        :param y_pred: Prediction
        :return: Mean Absolute Error; A measurement of prediction error
            - How far is our prediction from the true value?
        """

        return np.mean(np.abs(y_true - y_pred))

    # ================ PART 2 ================
    def poly_kfoldCV(self, x, y, p, K, seed=None) -> tuple:
        """
        :param x: training input (our x values)
        :param y: training output (our y values)
        :param p: degree of fitting polynomial (how many betas?)
        :param K: number of folds
        :return:
            train_error:    avg. MAE of training sets across all K folds
            cv_error:       avg. MAE of cross-validation sets across all K folds
        """

        # Step 0: Defensive shape handling (no variable-name changes)
        x = np.ravel(x)
        y = np.ravel(y)

        # Step 1.0: Create indices
        indices = x.shape[0]
        if K < 2 or K > indices:
            raise ValueError(f"K must be in [2, {indices}]")

        # Step 1.1: Local RNG for Reproducibility
        seed_generator = np.random.default_rng(seed)

        # Step 1.2: Reproducible index shuffling
        indices = seed_generator.permutation(indices)

        # Step 2.0: Split into K folds
        k_fold_subsets = np.array_split(indices, K)

        # Step 2.1: Create lists to store the error of each fold
        train_errors_kfold = []
        cross_validation_errors_kfold = []

        # Step 3: Loop through each fold
        for k_fold in range(K):
            # Extract validation subset
            val_idx = k_fold_subsets[k_fold]

            # Remaining indices are for training; Combine remainder of non-test data into training data
            train_idx = np.concatenate([k_fold_subsets[i] for i in range(K) if i != k_fold])

            # Separate data into training and validation sets
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            # Train the model on the training data
            # Find the best-fit polynomials (betas; beta_0, beta_1, etc.)
            betas = np.polyfit(x_train, y_train, p)

            # Make predictions on both training and validation sets
            y_train_prediction = np.polyval(betas, x_train)
            y_val_prediction = np.polyval(betas, x_val)

            # Calculate MAE for this fold's training and validation sets
            fold_train_error = self.mean_absolute_error(y_train, y_train_prediction)
            fold_cross_val_error = self.mean_absolute_error(y_val, y_val_prediction)

            # Add errors to the lists
            train_errors_kfold.append(fold_train_error)
            cross_validation_errors_kfold.append(fold_cross_val_error)

        # 4. Calculate the average error across all K folds
        train_error = np.mean(train_errors_kfold)
        cv_error = np.mean(cross_validation_errors_kfold)

        self._last_p = p
        self._last_train_error = float(train_error)
        self._last_cv_error = float(cv_error)

        return float(train_error), float(cv_error)


    # ---------- Plotting (matplotlib) ----------
    def sweep_polynomial(self, x, y, K, p_min=1, p_max=15, seed=None):
        """
        :param x: training input (our x values)
        :param y: training output (our y values)
        :param K: number of folds
        :param p_min: minimum p
        :param p_max: maximum p
        :param seed: random seed for reproducible folds across all p
        :return:
            ps: list of p values
            train_errors: list of training errors for each p
            cv_errors: list of cross-validation errors for each p
        """
        ps = list(range(p_min, p_max + 1))
        train_errors, cv_errors = [], []
        for p in ps:
            train_error, cv_error = self.poly_kfoldCV(x, y, p, K, seed=seed)
            train_errors.append(train_error)
            cv_errors.append(cv_error)
        return ps, train_errors, cv_errors

    @staticmethod
    def plot_errors(ps, train_errors, cv_errors, title="Training vs Cross-Validation Error", xlabel="p",
                    ylabel="Error"):
        """
        :param ps: list of p values
        :param train_errors: training errors aligned with ps
        :param cv_errors: cross-validation errors aligned with ps
        :param title: plot title
        :param xlabel: x-axis label
        :param ylabel: y-axis label
        :return: None
        """
        # Matplotlib kept strictly inside the class, as requested.
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(ps, train_errors, marker='o', label='Training Error')
        plt.plot(ps, cv_errors, marker='o', label='Cross-Validation Error')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ---------- Part 4, Learning Curves ----------
    def generate_learning_curves(self, x, y, p_values, K, N_min=20, N_max=100, N_step=5, seed=None):
        """
        Generates data for learning curves for specified polynomial degrees.

        :param x: Full training input
        :param y: Full training output
        :param p_values: A list of polynomial degrees to generate curves for
        :param K: Number of folds
        :param N_min: Minimum number of training samples
        :param N_max: Maximum number of training samples
        :param N_step: Step size for N
        :param seed: Random seed for reproducibility
        :return: A dictionary containing the results for each p.
                 Example: { p: {'Ns': [...], 'train_errors': [...], 'cv_errors': [...]}, ... }
        """
        # Ensure x and y are flat arrays
        x, y = np.ravel(x), np.ravel(y)

        N_values = list(range(N_min, N_max + 1, N_step))
        results = {}

        for p in p_values:
            train_errors_for_p = []
            cv_errors_for_p = []
            print(f"\n--- Generating Learning Curve Data for p={p} ---")

            for N in N_values:
                # Select the *first N samples* from the dataset
                x_subset, y_subset = x[:N], y[:N]

                train_error, cv_error = self.poly_kfoldCV(x_subset, y_subset, p, K, seed=seed)
                train_errors_for_p.append(train_error)
                cv_errors_for_p.append(cv_error)

            # Store the collected errors for the current p
            results[p] = {
                'Ns': N_values,
                'train_errors': train_errors_for_p,
                'cv_errors': cv_errors_for_p
            }

        return results


    # ---------- Standardized output helpers ----------
    @staticmethod
    def format_output(p, train_error, cv_error) -> str:
        """
        :param p: degree of fitting polynomial
        :param train_error: avg. MAE of training sets across all K folds
        :param cv_error: avg. MAE of cross-validation sets across all K folds
        :return: A standardized string for console output and logs
        """
        return (
            f"Current p Value {p}\n"
            f"train_error at p={p} = {train_error}\n"
            f"cv_error at p={p} = {cv_error}\n"
        )


    def __str__(self):
        """
        :return: Standardized string for the most recent result produced by poly_kfoldCV().
        """
        if self._last_p is None:
            return "No results have been computed yet."
        return self.format_output(self._last_p, self._last_train_error, self._last_cv_error)


def main():
    cv_lab = CrossValidationLab5()
    # Note: Using a fixed random seed for reproducibility of folds
    RANDOM_SEED = 42

    data = pd.read_csv("./data/data_lab5.csv")
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    # ================ PART 3 ================
    print("=" * 20 + " PART 3: MODEL SELECTION " + "=" * 20)
    p1_train, p1_cv = cv_lab.poly_kfoldCV(x, y, p=1, K=5, seed=RANDOM_SEED)
    print("Verification for p=1, K=5:")
    print(cv_lab.format_output(1, p1_train, p1_cv))

    # Sweep p = [1..15] with the SAME folds for fair comparison, then plot
    ps, train_errors, cv_errors = cv_lab.sweep_polynomial(x, y, K=5, p_min=1, p_max=15, seed=42)
    cv_lab.plot_errors(ps, train_errors, cv_errors,
                       title="5-Fold CV (K=5): Training vs Cross-Validation Error",
                       xlabel="Polynomial degree p",
                       ylabel="Mean Absolute Error (MAE)")

    # Find and print the best p from the sweep
    best_p_index = np.argmin(cv_errors)
    best_p = ps[best_p_index]
    min_cv_error = cv_errors[best_p_index]

    # ================ PART 3 STEP 5 ANSWER ================
    print("\n--- Part 3: Model Selection Answer ---")
    print(
        f"The best model complexity is p={best_p}, which has the lowest cross-validation error of {min_cv_error:.4f}.")
    print("Models with p > 5 show decreasing training error but increasing CV error, indicating overfitting.\n")


    # ================ PART 4 ================
    print("\n" + "=" * 20 + " PART 4: LEARNING CURVES " + "=" * 20)


    # ============================================================
    # NOTE TO SELF: Run this on its own and check out the warnings
    p_values_part4 = [1, 2, 7, 10, 16]
    learning_curve_results = cv_lab.generate_learning_curves(
        x, y, p_values=p_values_part4, K=5, seed=RANDOM_SEED
    )
    # What's happening:
    #   NumPy is telling us that there are too few points for the power/complexity of the model.
    #   The model is becoming *unstable*
    #   Attempting to fit a complex polynomial (e.g., p = 16 has 17 betas) to a small number of data points
    #           e.g., N=20 or 25
    #
    # **Severe overfitting**
    #
    # Analogy: Drawing a unique line hitting 20 dots. Line will be convoluted and any change will have drastic results
    #
    # ============================================================


    # Plot the learning curve for each value of p
    for p, results in learning_curve_results.items():
        cv_lab.plot_errors(
            ps=results['Ns'],  # The x-axis is now N, not p
            train_errors=results['train_errors'],
            cv_errors=results['cv_errors'],
            title=f"Learning Curve for Polynomial Degree p={p}",
            xlabel="Number of Training Samples (N)",
            ylabel="Mean Absolute Error (MAE)"
        )


    # ================ PART 4: ANALYSIS ANSWERS ================
    print("\n" + "=" * 20 + " PART 4: ANALYSIS ANSWERS " + "=" * 20)
    print("\n--- 3a) Highest Bias ---")
    print(
        "The model with p=1 has the highest bias. Its learning curve shows both training and Cross Validation errors "
        "converging at a high value. This means the model is too simple (underfitting) and cannot capture the "
        "data's complexity, even with more samples."
        "It will have predictably inaccurate results on unseen data"
    )

    print("\n--- 3b) Highest Variance ---")
    print(
        "The model with p=16 has the highest variance. Its learning curve shows a very low training error but a very "
        "high CV error, with a large gap between them. This indicates the model is too complex (overfitting) and is "
        "memorizing noise in the training data instead of generalizing."
        "It is likely to perform poorly on unseen data."
    )

    print("\n--- 4a) Best model for N=50 samples ---")
    print(
        "By inspecting the plots at N=50, the model with p=7 appears to be the best choice. It offers one of the lowest "
        "cross-validation errors at this sample size, providing a balance between bias and variance. The simpler models "
        "(p=1, p=2) have higher errors (higher bias), and the more complex models (p=10, p=16) have similar or higher "
        "Cross Validation errors due to overfitting a small dataset."
    )

    print("\n--- 4b) Best model for N=80 samples ---")
    print(
        "With 80 samples, the model with p=7 still appears to be the strongest choice. At N=80, its cross-validation "
        "error is the lowest among the options. The additional data helps reduce the variance of the more complex "
        "models, but p=7 provides the most effective generalization for this dataset size."
    )


if __name__ == "__main__":
    main()