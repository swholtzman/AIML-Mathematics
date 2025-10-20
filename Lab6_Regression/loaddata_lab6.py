import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load and encode categorical data
# Inputs:
#   filepath: path to the csv file
# Outputs:
#   data: a DataFrame object containing encode data

class LoadData:

    @staticmethod
    def one_hot(data: pd.DataFrame) -> pd.DataFrame:
        """
        If a column is dtype 'object' OR has <= 20 unique values,
        expand it into one-hot dummy columns (0/1), preserving
        the original attribute order.

        :param data: dataframe of csv file contents with dropna()
        :return: one-hot encoded dataframe
        """

        # snapshot original columns so we don't iterate
        #   over newly created dummy columns
        columns = list(data.columns)

        for column in columns:
            if (data[column].dtype == 'object') or (data[column].nunique(dropna=True) <= 20):
                dummies = pd.get_dummies(data[column], prefix=column, dtype='uint8')
                index = data.columns.get_loc(column)
                left = data.columns[:index]
                right = data.columns[index+1:]

                # replace column with its dummies at the same position
                data = pd.concat([data[left], dummies, data[right]], axis=1)

        return data

    def load(self, filename: str) -> pd.DataFrame:
        """
        Intake csv file and return a pandas DataFrame

        :param filename: csv file name
        :return: pandas DataFrame
        """
        data = pd.read_csv(filename, low_memory=False).dropna()
        data = self.one_hot(data)
        return data


def main():
    data_loader = LoadData()
    data = data_loader.load(filename="./data/data_lab6.csv")
    data_expected = pd.read_csv("./data/data_lab6_expanded.csv", low_memory=False)
    print(np.allclose(data, data_expected))


if __name__ == "__main__":
    main()