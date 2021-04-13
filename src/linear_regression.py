import warnings

import numpy as np

from linear import Linear


class LinearRegression(Linear):

    def __init__(self, x, y, alpha: float = 0.1, iterations: int = 1400, algorithm: str = "batch"):

        super().__init__(x, y, alpha, iterations)
        np.seterr(all='warn')

        # Defining the algorithm
        self.__algorithm = algorithm
        assert self.__algorithm in ["batch", "normal"], "Algorithms available are 'batch' and 'normal', " \
                                                        "which refer to batch gradient descent and normal" \
                                                        "equation algorithms, respectively"

    def hypothesis(self):
        """Return the value of the hypothesis based on X and theta"""
        return np.dot(self._X, self._theta)

    def cost_function(self):
        """Calculates the cost function. J(theta) = (1/2m)* E (h(x) - y)^2"""

        h = self.hypothesis()

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                E = sum(np.power(h - self._y, 2))
            except RuntimeWarning:
                raise RuntimeError("The values might be too big for an operation. Try choosing a smaller alpha.")

        return E / (2 * self._m)

    def normal_equation(self):
        """Calculates theta through the normal equation"""

        self._theta = np.linalg.pinv(np.dot(self._X.T, self._X)) * self._X.T * self._y

    def fit(self):

        if self.__algorithm == "batch":
            self.gradient_descent()

        elif self.__algorithm == "normal":
            self.normal_equation()
