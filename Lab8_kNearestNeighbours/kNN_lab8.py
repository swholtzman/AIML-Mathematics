import math
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score

class kNN:
    def __init__(self, X_data=None, Y_data=None, k=3, full_set=None, k_values=None):
        self.X_data = X_data
        self.Y_data = Y_data
        self.k = k
        self.full_set = full_set
        self.k_values = k_values
        self.avg_error_rates = []
        self.lowest_k = Any


    # Helper for Euclidean Distances
    @staticmethod
    def calculate_euclidean_distance(x_const, x_trial) -> float:
        return math.sqrt( (x_const[0] - x_trial[0])**2 + (x_const[1] - x_trial[1])**2 )

    # k-Nearest Neighbours
    # Inputs:
    #   x: a test sample
    #   Xtrain: array of input vectors of the training set
    #   Ytrain: array of output values of the training set
    #   k: number of nearest neighbours
    # Outputs:
    #   y_pred: predicted value for the test sample
    def kNN(self, x, Xtrain, Ytrain, k) -> int:
        all_distances = []
        y_pred = 0

        # ---------- Step 1: Calculate Distances ----------
        for i in range(len(Xtrain)):
            # get the current training sample
            train_sample = Xtrain[i]

            # calculate its distance from the test sample x
            distance = self.calculate_euclidean_distance(x, train_sample)

            # store distance AND class label of that training sample
            all_distances.append( (distance, Ytrain[i]) )
        # ---------- ---------------------------- ----------


        # ---------- Step 2: Sort and Select Neighbours ----------
        all_distances.sort()
        # ---------- ---------------------------------- ----------

        # ---------- Step 3: Majority Vote ----------
        votes = {}
        # look at only the first 'k' (nearest) neighbours
        for i in range(k):
            neighbour_class = all_distances[i][1]

            # add to the vote count
            votes[neighbour_class] = votes.get(neighbour_class, 0) + 1

        # find the class with the most votes
        # max(votes, key=votes.get) finds key w/ the max value
        y_pred = max(votes, key=votes.get)
        # ---------- --------------------- ----------

        # ---------- Step 4: Return prediction ----------
        return y_pred

    def verify(self) -> float:
        # Initialize holders
        errors = 0

        # loop must run for all samples , not just k
        for i in range(len(self.X_data)):

            # Validation Set (The left out set)
            x = self.X_data[i]
            actual_y = self.Y_data[i]

            # Training Set (Everything that is not in the left out set; All samples except i)
            Xtrain = [self.X_data[j] for j in range(len(self.X_data)) if j != i]
            Ytrain = [self.Y_data[j] for j in range(len(self.Y_data)) if j != i]

            # call the kNN function
            y_pred = self.kNN(x, Xtrain, Ytrain, k=self.k)
            print(f"Trial {i + 1} (Validate x{i + 1}): Actual={actual_y}, Predicted={y_pred}", end="")

            # Compare y_pred to Y_data[i]
            if self.Y_data[i] != y_pred:
                errors += 1
                print(" -> Error")
            else:
                print(" -> Correct")

        # Calculate Average Error
        error_rate = errors / len(self.X_data)
        print(f"\nTotal Errors: {errors}")
        print(f"Average LOOCV Error: {error_rate * 100:.2f}%")

        return error_rate
    # ----------------------------------------------------------------------

    # PART 2
    def kNN_sklearn(self) -> float:
        X_np = np.array(self.X_data)
        Y_np = np.array(self.Y_data)

        # 1. Initialize the LeaveOneOut (LOOCV) splitter
        loocv = LeaveOneOut()

        # 2. Initialize the sklearn classifier
        sklearn_knn = KNeighborsClassifier(n_neighbors=self.k)

        errors = 0
        trial = 1

        # 3. Loop through each LOOCV split
        # 'train_index' and 'test_index' are lists of indices
        for train_index, test_index in loocv.split(X_np):
            # get training and test validation sets for this trial
            X_train, X_test = X_np[train_index], X_np[test_index]
            Y_train, Y_test = Y_np[train_index], Y_np[test_index]

            # 4. fit the model on the 4 training samples
            sklearn_knn.fit(X_train, Y_train)

            # 5. predict the class for the 1 validation sample
            y_pred = sklearn_knn.predict(X_test)

            # 6. check for error
            actual_y = Y_test[0]
            predicted_y = y_pred[0]

            print(f"Trial {trial}: Actual={actual_y}, Predicted={predicted_y}", end="")
            if actual_y != predicted_y:
                errors += 1
                print(" -> Error")
            else:
                print(" -> Correct")

            trial += 1

        # Calculate Average Error
        error_rate = errors / len(self.X_data)
        print(f"\nTotal Errors: {errors}")
        print(f"Average LOOCV Error: {error_rate * 100:.2f}%")

        return error_rate
    # ----------------------------------------------------------------------

    # PART 3
    def pre_processing(self) -> tuple[Any, Any, Any, Any]:
        # Separate features into Features (x) and Labels (y)
        X_data = self.full_set.iloc[:, :-1]
        Y_data = self.full_set.iloc[:, -1]

        # split X_data into training (75%) and test (25%) sets
        X_train = X_data[:(int(len(X_data) * 0.75))]
        X_test = X_data[(int(len(X_data) * 0.75)):]

        # split Y_data into training (75%) and test (25%) sets
        Y_train = Y_data[:(int(len(Y_data) * 0.75))]
        Y_test = Y_data[(int(len(Y_data) * 0.75)):]

        return X_train, X_test, Y_train, Y_test

    def cross_validation(self, X, y, cv):
        k_values = self.k_values
        for k in k_values:
            classifier = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(classifier, X, y, cv=cv)
            accuracy = scores.mean()
            error_rate = 1 -  accuracy

            self.avg_error_rates.append(error_rate)

    def create_plot(self):
        plt.plot(self.k_values, self.avg_error_rates, marker='o', label='CV Error Rate')
        plt.xlabel("Values of k")
        plt.ylabel("Average Error Rates")
        plt.title("Average Cross Validation Estimate of Prediction Error as a Function of k")

        plt.xticks(self.k_values)  # Set x-axis ticks to k-values
        plt.grid(True)  # Add a grid

        plt.legend()  # This will automatically pick up the labels
        plt.savefig("kNN_lab8.png")
        plt.show()

    def evaluate_on_test_set(self, X_train, Y_train, X_test, Y_test) -> float:
        error_count = 0
        # Initializes a KNeighborsClassifier using self.lowest_k.
        evaluation_model = KNeighborsClassifier(n_neighbors=self.lowest_k)
        # Fit the classifier on X_train and Y_train
        evaluation_model.fit(X_train, Y_train)
        # uses classifier to predict labels for X_test
        y_pred = evaluation_model.predict(X_test)

        print(f"\n--- Test Set Evaluation (k={self.lowest_k}) ---")
        # compare those to actual labels in Y_test to count the errors and find the error rate
        for prediction, actual in zip(y_pred, Y_test.values):
            if prediction != actual:
                error_count += 1
                print(f"Error:   Actual={actual}, Predicted={prediction}")
            else:
                print(f"Correct: Actual={actual}, Predicted={prediction}")

        # Calculate and print the final error rate for the test set
        test_error_rate = 0.0
        if len(Y_test) > 0:
            test_error_rate = error_count / len(Y_test)

        print(f"\nTotal Test Errors: {error_count} / {len(Y_test)}")
        print(f"Test Set Error Rate: {test_error_rate * 100:.2f}%")

        # Return the error rate (float), not the count
        return test_error_rate

    def run_kNN(self, cv):
        X_train, X_test, Y_train, Y_test = self.pre_processing()
        self.cross_validation(X_train, Y_train, cv)
        self.create_plot()

        lowest_error = min(self.avg_error_rates)
        err_index = self.avg_error_rates.index(lowest_error)
        self.lowest_k = self.k_values[err_index]

        self.evaluate_on_test_set(X_train, Y_train, X_test, Y_test)
    # ----------------------------------------------------------------------

def main():
    # Step 1: Define Data
    X_data = [[1,1], [2,3], [3,2], [3,4], [2,5]]
    Y_data = [0, 0, 0, 1, 1]

    # --- PART 1: Run Custom kNN function
    print("--- Verifying custom kNN function ---")
    my_classifier = kNN(X_data, Y_data, k=3)
    my_classifier.verify()
    print("\n" + "="*40 + "\n")

    # --- PART 2: Run sklearn Verification ---
    print("--- Verifying with sklearn ---")
    my_classifier.kNN_sklearn()

    # --- PART 3: Test on csv file
    # load data from csv file with pandas
    csv_file_path = "data_lab8.csv"
    # df == DataFrame
    df = pd.read_csv(csv_file_path)

    k_values = [k for k in range(1,22,2)]
    csv_classifier = kNN(full_set=df, k_values=k_values)
    csv_classifier.run_kNN(cv=10)



if __name__ == "__main__":
    main()