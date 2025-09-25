from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

# provided generator + ground-truth function:
#   f(x) = x + sin(3x)
#   genNoisyData() returns a shuffled x array (50 pts in [0,6]) and y = f(x)+N(0,0.25^2)
from GenData_Lab4 import genNoisyData, f  # <- provided by the lab

# -----------------------------
# configuration
# -----------------------------
EVALUATION_X: float = 5.0         # given x = 5,...
NUM_DATASETS: int = 1000          # given 1000 sets,...
DEGREES: List[int] = [1, 3, 5, 9, 15] # p values
NOISE_SIGMA: float = 0.25         # given stdev = 0.25
NOISE_VAR: float = NOISE_SIGMA ** 2          # <-- for expected MSE
RANDOM_SEED: int | None = 42      # set to None for true randomness; 42 for consistency


@dataclass
class PointwiseStats:
    """Holds bias/variance results for one polynomial degree p where x = 5."""
    degree: int
    bias: float
    variance: float
    # inspect later:
    mean_prediction: float
    true_value: float

    bias2: float
    expected_mse: float


class PolynomialModel:
    """
    Wraps a stable polynomial fit:
      - fit() learns coefficients using numpy.polynomial.Polynomial.fit
      - predict_at(x0) evaluates fitted poly at a single x value (or array)
    """
    def __init__(self, degree: int):
        self.degree = int(degree)
        self._model = None
        self._domain = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit poly model to the (x_train, y_train) data
        Using Polynomial.fit maps x to [-1,1] internally -> numerically stable
        """
        from numpy.polynomial import Polynomial as Poly
        self._domain = [float(np.min(x_train)), float(np.max(x_train))]
        self._model = Poly.fit(x_train, y_train, deg = self.degree, domain = self._domain)

    def predict_at(self, x_eval: float | np.ndarray) -> np.ndarray:
        """
        Evaluate fitted poly model at x_eval
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit(...) first.")
        return self._model(x_eval)


class BiasVarianceRunner:
    """
      • Generate NUM_DATASETS datasets
      • For each degree p, fit to each dataset and predict at x = 5
      • Compute Bias and Variance at x = 5
      • Plot histogram for each p
      • Report smallest |bias|, lowest variance, and goldilocks (min expected MSE)

    """
    def __init__(self, degrees: List[int], num_datasets: int,
                 evaluation_x: float, random_seed: int | None = None):

        self.degrees = list(degrees)
        self.num_datasets = int(num_datasets)
        self.x0 = float(evaluation_x)
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Precompute true function value at x = 5 (used for bias)
        self.true_value_at_x0: float = float(f(self.x0))

    def _collect_predictions_for_degree(self, degree: int) -> np.ndarray:
        """
        For each dataset m:
          - generate data (x_m, y_m)
          - fit degree-p polynomial
          - predict at x = 5
        Return vector of 1000 predictions (shape: [num_datasets])
        """

        predictions = []
        for _ in range(self.num_datasets):
            x_obs, y_obs = genNoisyData()  # 50 samples in [0,6] with noise
            model = PolynomialModel(degree)
            model.fit(x_obs, y_obs)
            y_hat_at_x0 = float(model.predict_at(self.x0))  # scalar prediction
            predictions.append(y_hat_at_x0)
        return np.asarray(predictions)

    def _compute_pointwise_stats(self, degree: int, predictions: np.ndarray) -> PointwiseStats:
        """
        Compute bias and variance at x = 5 for a given degree
          • Bias(p) = mean_yhat - f(5)
          • Var(p)  = variance of predictions across datasets
        """
        mean_pred = float(np.mean(predictions))
        bias = mean_pred - self.true_value_at_x0
        variance = float(np.var(predictions, ddof = 0))
        bias2 = float(bias ** 2)
        expected_mse = float(bias2 + variance + NOISE_VAR)
        return PointwiseStats(
            degree = degree,
            bias = float(bias),
            variance = variance,
            mean_prediction = mean_pred,
            true_value = self.true_value_at_x0,
            bias2 = bias2,
            expected_mse = expected_mse,
        )

    def _plot_histogram(self, degree: int, predictions: np.ndarray, stats: PointwiseStats) -> None:
        """
        Plot histogram of 1000 predictions at x=5 for given degree p
        Adds vertical lines for mean prediction (red) and f(5) (black)
        """
        plt.figure()
        plt.hist(predictions, bins = 30)
        plt.axvline(stats.mean_prediction, color = "red", linewidth = 2, label = "mean of ŷ(x=5)")
        plt.axvline(stats.true_value, linewidth = 2, label = "f(x=5)")
        plt.title(f"Histogram for p = {degree}")
        plt.xlabel("ŷ_pred(x = 5)")
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()
        # Show plot:
        # plt.savefig(f"hist_p{degree}.png", dpi=120)
        plt.show()

    def run(self, make_plots: bool = True) -> List[PointwiseStats]:
        """
        Execute full experiment for all degrees and make plots
        """
        results: List[PointwiseStats] = []

        for p in self.degrees:
            predictions = self._collect_predictions_for_degree(p)
            stats = self._compute_pointwise_stats(p, predictions)
            results.append(stats)

            # Print in  required format
            print(f"For p = {p}:")
            print(f"    Bias Measure \t\t= {stats.bias}")
            print(f"    Variance Measure \t= {stats.variance}")

            if make_plots:
                self._plot_histogram(p, predictions, stats)

        # Identify smallest abs. bias, lowest variance, and goldilocks zone
        smallest_bias_stats = min(results, key = lambda r: abs(r.bias))
        lowest_var_stats = min(results, key = lambda r: r.variance)
        goldilocks_stats = min(results, key = lambda r: r.expected_mse)


        print("\n")
        print(f"Lowest Bias Point: p = {smallest_bias_stats.degree}  "
              f"\n\tWhere Bias      = {smallest_bias_stats.bias}")
        print(f"Lowest Variance Point: p = {lowest_var_stats.degree}  "
              f"\n\tWhere Variance  = {lowest_var_stats.variance}")
        print(f"Goldilocks Zone (min expected MSE): p = {goldilocks_stats.degree}  "
              f"\n\tBias^2 + Var + σ^2 = {goldilocks_stats.expected_mse}")
        print("")

        return results


def main() -> None:
    runner = BiasVarianceRunner(
        degrees=DEGREES,
        num_datasets=NUM_DATASETS,
        evaluation_x=EVALUATION_X,
        random_seed=RANDOM_SEED,   # set to None to see different numbers each run
    )
    runner.run(make_plots=True)


if __name__ == "__main__":
    main()
