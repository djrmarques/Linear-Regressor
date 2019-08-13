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

    def __init__(self, X_train, y ):
        """ 
        X_train -> n_samples * n_features
        y -> n_samples 
        """

        self.r2 = None
        self.prediction = None

        self.rss = None
        self.tss = None

        # Save the label and the training set
        self._inputs_check(X_train, y)
        self.y = y
        self.X_train = X_train

    def fit(self):
        """ Determined the coeficients for the regression """

        # Concatenate values of 1 in the array
        X_train = np.concatenate((np.ones((1, self.X_train.shape[1])), self.X_train), axis=0)

        XT_X = np.matmul(X_train, X_train.T)

        # This happens in a simple linear regression
        if X_train.shape[1] == 1:
            inv_XT_X = np.array(1/XT_X).reshape(1, 1)
        else:
            inv_XT_X = np.linalg.inv(XT_X)

        invXTX_XT = np.matmul(inv_XT_X, X_train)
        coefs = np.matmul(invXTX_XT, self.y.T).flatten()

        # Save the coeficients and calculate the score
        self.coefs = np.reshape(coefs, (-1, 1))
        self.score()


    def score(self):
        """ Scores the obtained on the provided test set. Uses R squared."""

        # Initial assertions
        assert self.coefs is not None, "Run the fit method first."
        X_train = np.concatenate((np.ones((1, self.X_train.shape[1])), self.X_train), axis=0)

        # Evaluate the X_test points
        self.prediction = np.sum(X_train * self.coefs, axis=0)

        self.rss = np.square(self.y - self.prediction).sum()
        self.tss = np.square(self.y - self.y.mean()).sum()
        self.r2 = 1 - (self.rss/self.tss)
        assert 0 <= self.r2 <= 1, f"R2 square must be between 0 and 1 not {self.r2}"
        print(f"R2 square: {self.r2}")

    def regression_analysis(self):
        """ Uses statsmodels to print the analysis of the regression variables"""

        X = sm.add_constant(self.X_train.T)
        mod = sm.OLS(self.y.T, X)
        self.res = mod.fit()
        print(self.res.summary())

    def select_features(self):
        """ Performs feature selection """

        original_X = self.X_train.copy()
        n_features = self.X_train.shape[0]
        features_list = list(range(n_features))
        current_features = list()
        best_r2 = 0
        self.X_train = None

        while features_list:
            best_feature = None
            for feature in features_list:
                if not self.X_train:
                    self.X_train = original_X[:, feature].reshape(1, -1)
                else:
                    self.X_train[:, -1] = original_X[:, feature]
                self.fit()
                if self.r2 > best_r2:
                    best_r2 = self.r2
                    best_feature = feature

            if best_feature:
                self.X_train = original_X[:, feature]

            del features_list[best_feature]

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
        assert y.shape[0] == 1, f"y array should have dimention one, not {y.dim}"
        assert X.shape[1] == y.shape[1], "The dependent and independt variables do not have the same shape."

# The parentehsis are just because of EMACS
if (__name__ == "__main__"):

    import pandas as pd

    datapath = "../islr/datasets/Advertising.csv"
    df = pd.read_csv(datapath, index_col=0)

    X = df.drop("sales", axis=1).values.T
    # X = df["TV"].values.reshape(1, -1)
    y = df["sales"].values
    y = y.reshape(1, y.shape[0])
    lr = LinearRegression(X, y)
    lr.select_features()
