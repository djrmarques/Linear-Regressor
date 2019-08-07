#!/usr/bin/env python3
# This module performs linear regression on the provided data presents the relevant statistics scores
# This module was written while following the ISLR book

import numpy as np
import statsmodels.api as sm


class LinearRegression:
    """
    Linear Regression.  This is class is not meant to be used in production, it
    was just designed as a learning tool.
    """

    def __init__(self):
        self.coeficients = None
        self.r2 = None
        self._polyfit = None
        self.prediction = None

        self.rss = None
        self.tss = None

        self.y = None
        self.X_train = None


    def fit(self, X_train, y):
        """ Determined the coeficients for the regression """

        # Check the input formats
        self._inputs_check(X_train, y)

        # Save the label and the training set
        self.y = y
        self.X_train = X_train

        # Reshape the variables
        y = y.reshape(1, y.shape[0])
        X_train = X_train.reshape(-1, X_train.shape[0])

        # Concatenate values of 1 in the array
        X_train = np.concatenate((X_train, np.ones((1, X_train.shape[1]))), axis=0)

        XT_X = np.matmul(X_train, X_train.T)

        # This happens in a simple linear regression
        if X_train.shape[1] == 1:
            inv_XT_X = np.array(1/XT_X).reshape(1, 1)
        else:
            inv_XT_X = np.linalg.inv(XT_X)

        invXTX_XT = np.matmul(inv_XT_X, X_train)
        coefs = np.matmul(invXTX_XT, y.T).flatten()

        # Save the coeficients
        self.coefs = coefs
        self._polyfit = np.poly1d(coefs)

    def score(self, X_test):
        """ Scores the obtained on the provided test set. Uses R squared."""

        assert self.coefs is not None, "Run the fit method first."

        # Convert array to 2D, if it is only 1D.
        if X_test.ndim > 1:
            X_test = X_test.T

        # Evaluate the X_test points
        self.prediction = np.apply_along_axis(self._polyfit, axis=0, arr=X_test)

        self.rss = np.square(self.y - self.prediction).sum()
        self.tss = np.square(self.y - self.y.mean()).sum()
        self.r2 = 1 - self.rss/self.tss

    def feature_selection(self):
        pass

    def regression_analysis(self):
        """ Uses statsmodels to print the analysis of the regression variables"""

        X = sm.add_constant(self.X_train)
        mod = sm.OLS(self.y, X)
        res = mod.fit()
        print(res.summary())

    @staticmethod
    def _inputs_check(X_train, y):
        """ Checks if the provided inputs are in the right format """

        # Assert that both X, y are ndarrays
        assert type(X_train) == np.ndarray, "Please pass X as a np.ndarray"
        assert type(y) == np.ndarray, "Please pass y as a np.ndarray"

        # Check that there are no nans
        assert np.isnan(X_train).sum() == 0, "There are missing values in X"
        assert np.isnan(y).sum() == 0, "There are missing values in y"

        # Check the shape of the inputs
        assert y.ndim == 1, f"y array should have dimention one, not {y.dim}"
        assert X.shape[0] == y.shape[0], "The dependent and independt variables do not have the same shape."

# The parentehsis are just because of EMACS
if (__name__ == "__main__"):

    import pandas as pd

    datapath = "../islr/datasets/Advertising.csv"
    df = pd.read_csv(datapath, index_col=0)

    X = df.drop("sales", axis=1).values
    y = df["sales"].values

    lr = LinearRegression()
    lr.fit(X, y)
    lr.regression_analysis()
