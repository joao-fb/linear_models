# Linear Models
Linear Models in Python based on Andrew Ng's Machine Learning course.

It accepts both numpy ndarrays and pandas Data Frames/Series. You should import either LinearRegression or LogisticRegression.

To create a model, pass the design matrix X as a np ndarray or a Data Frame to the 'x' parameter in the constructor. Pass the true value/label
to the y parameter in the constructor. You should also choose the right alpha and number of iterations. Otherwise, the Cost Function values may have
an adverse effect: either gradient descent won't converge, or its values will become too small.

For Linear Regression you can also use the Normal Equation to fit the model (only recommended when you have less than 10k traning examples).

After choosing the constructor parameters, you should fit the model using the method 'fit'. Then, you can predict new values using the 'predict' method, or by
accessing the attribute 'theta'. 'theta' provides the coefficients for each feature (including a bias column with only ones (1)).
