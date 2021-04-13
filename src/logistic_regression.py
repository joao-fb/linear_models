import numpy as np

from linear import Linear


class LogisticRegression(Linear):

    def __init__(self, x, y, alpha: float = 0.1, iterations: int = 1400):

        super().__init__(x, y, alpha, iterations)

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.power(np.e, -z))

    def hypothesis(self):
        return self.sigmoid(np.dot(self._X, self._theta))

    def cost_function(self):
        h = self.hypothesis()

        # When y is equal to 1
        y1 = np.dot(np.log(h), self._y)

        # When y is equal to 2
        y0 = np.dot(np.log(h), (1 - self._y))

        return (1/self._m)*(-y1 - y0)

    def fit(self):
        self.gradient_descent()
