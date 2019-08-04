#!/usr/bin/env python3
# This module performs linear regression on the provided data presents the relevant statistics scores
# This module was written while following the ISLR book
import numpy as np

class LinearRegression:
    """ Linear Regression. This is class is not meant to be used in production, it was just designed as a learning tool. """

    def fit(self, X, y):
        """ Determined the coeficients for the regression """
        pass

    @staticmethod
    def _inputs_check(X, y):
        """ Checks if the provided inputs are in the right format """

        # Assert that both X, y are ndarrays
        assert type(X) == np.ndarray, "Please pass X as a np.ndarray"
        assert type(y) == np.ndarray, "Please pass y as a np.ndarray"

        # Check that there are no nans
        assert np.isnan(X).sum() == 0, "There are missing values in X"
        assert np.isnan(y).sum() == 0, "There are missing values in y"

        # Check the shape of the inputs
        assert X.shape[0] == y.shape[0], "The dependent and independt variables do not have the same shape."

    def _single_fit(self, X, y):
        """ This method fits a simple linear regression (Only one indepent variable)"""

        self._inputs_check(X, y)

        # Assert that the training set only has one variable
        n_independent = X.shape[1]
        assert n_independent == 1, f"The number of indepent variables is {n_independent} and should be just one."


    def predict(self, X):
        """ Makes predictions on a test data set"""
        pass

    def fit_single_linear_regression(self):
        """ Performs a fit method"""
        pass


# The parentehsis are just because of EMACS
if (__name__ == "__main__"):

    import pandas as pd

    datapath = "../islr/datasets/Advertising.csv"
    df = pd.read_csv(datapath, index_col=0)
    print(df.head())

    X = df.drop("sales", axis=1).values
    y = df["sales"].values

    lr = LinearRegression()
    lr.fit(X, y)
