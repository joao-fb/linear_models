from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd


class Linear(ABC):

    def __init__(self, x, y, alpha: float = 0.1, iterations: int = 1400):
        np.seterr(all='warn')

        # Defining alpha and iterations
        self._alpha = alpha
        self._iterations = iterations

        # Checking alpha and iterations
        assert isinstance(self._alpha, float) or isinstance(self._alpha, int), "Alpha must be a float or integer"
        assert isinstance(self._iterations, int), "The number of iterations performed must be an integer"

        # Treating matrices
        self._X = self.treat(x)
        self._y = self.treat(y)

        # Creating m and n attributes. M refer to the number of trainings,
        # N refers to the number of features
        try:
            self._m = self._X.shape[0]
            self._n = self._X.shape[1]
        except IndexError:
            # If X is only a vector, we need to transform it into a matrix
            self._X = np.array([self._X]).T
            self._n = self._X.shape[1]

        # Adding column of ones to the design matrix

        ones = np.array([np.ones(self._m)])
        self._X = np.append(ones.T, self._X, axis=1)

        # Creating N + 1 dimensional vector for theta (number of features and linear coefficient)
        self._theta = np.zeros(self._n + 1)

        # Creating a vector for the historical values of the cost function
        self._j_history = np.empty([0, 2])

        # Checks for the design matrix
        assert isinstance(self._X, np.ndarray), "Design matrix X must be a list, DataFrame, " \
                                                "or numpy n dimensional array"

        # Checks for the vector we want to predict
        assert isinstance(self._y, np.ndarray), "Y must be a list, DataFrame, \
                                                        or numpy n dimensional array"
        assert len(self._y.shape) == 1, "Y must be a vector"

    @staticmethod
    def treat(matrix):
        """Transforms matrix into numpy ndarray if matrix is
        instance of DataFrame, list, or already an ndarray."""

        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.to_numpy()

        elif isinstance(matrix, pd.Series):
            matrix = matrix.to_numpy()

        elif isinstance(matrix, np.ndarray):
            matrix = matrix

        elif isinstance(matrix, list):
            matrix = np.array(matrix)

        else:
            matrix = None

        return matrix

    @abstractmethod
    def hypothesis(self):
        pass

    @abstractmethod
    def cost_function(self):
        pass

    def gradient_descent(self):
        """Calculates theta through batch gradient descent"""

        for i in range(self._iterations):

            h = self.hypothesis()

            E = np.dot(h - self._y, self._X)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    self._theta = self._theta - (self._alpha/self._m) * E
                except Warning:
                    raise RuntimeError("Choose a smaller alpha for the iterations.")

            # Inserts the number of the iteration and the value of the cost function in the i
            # index of the array
            self._j_history = np.append(self._j_history, [[i, self.cost_function()]], axis=0)

    @abstractmethod
    def fit(self):
        pass

    def predict(self, x_test):
        x_test = self.treat(x_test)

        m_test = x_test.shape[0]
        n_test = x_test.shape[1]

        assert n_test == self._n, """The matrix you want to predict must have the
                                            same number of columns as the design matrix. """

        # Creates linear coefficient column
        ones = np.array([np.ones(m_test)])
        x_test = np.append(ones.T, x_test, axis=1)

        return np.dot(x_test, self._theta)

    @property
    def j_history(self):
        return self._j_history

    @property
    def theta(self):
        return self._theta
