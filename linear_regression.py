#!/usr/bin/env python3
# This module performs linear regression on the provided data presents the relevant statistics scores
# This module was written while following the ISLR book

import numpy as np
import matplotlib.pyplot as plt


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

        self.label = None


    def fit(self, X, y):
        """ Determined the coeficients for the regression """
        pass

    def _single_fit(self, X_train, y):
        """ This method fits a simple linear regression, and gets the parameters (Only one indepent variable) """

        self._inputs_check(X_train, y)

        self.y = y  # TODO: Change this later
        self.X = X  # TODO: Change this later

        # Assert that the training set only has one variable
        n_independent = X_train.ndim
        assert n_independent == 1, f"The number of indepent variables is {n_independent} and should be just one."

        X_train_mean = X.mean()
        y_mean = y.mean()

        beta1 = ((X_train - X_train_mean)*(y - y_mean)).sum()/(np.sum(np.square(X-X_train_mean)))
        beta0 = y_mean - beta1*X_train_mean

        # Save the coeficients
        self.coeficients = np.array([beta1, beta0])
        self._polyfit = np.poly1d(self.coeficients)

    def scatter_matrix(self):
        """ Plots the scatter matrix with the corresponding linear fit line """

        plt.figure(figsize=(10, 10))

        plt.scatter(self.X, self.y, alpha=0.5)
        plt.plot(tuple(map(self._polyfit, range(0, self.X.max().astype(int)))), color="red")
        plt.show()


    def score(self, X_test):
        """ Scores the obtained on the provided test set. Uses R squared."""

        assert self.coeficients is not None, "Run the fit method first."

        # Convert array to 2D, if it is only 1D.
        if X_test.ndim > 1:
            X_test = X_test.T

        # Evaluate the X_test points
        self.prediction = np.apply_along_axis(self._polyfit, axis=0, arr=X_test)

        self.rss = np.square(self.y - self.prediction).sum()
        self.tss = np.square(self.y - self.y.mean()).sum()
        self.r2 = 1 - self.rss/self.tss

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
        assert y.ndim == 1, f"y array should have dimention one, not {y.dim}"
        assert X.shape[0] == y.shape[0], "The dependent and independt variables do not have the same shape."

# The parentehsis are just because of EMACS
if (__name__ == "__main__"):

    import pandas as pd

    datapath = "../islr/datasets/Advertising.csv"
    df = pd.read_csv(datapath, index_col=0)

    X = df["TV"].values
    y = df["sales"].values

    lr = LinearRegression()
    lr._single_fit(X, y)
    lr.score(X)
    lr.scatter_matrix()
