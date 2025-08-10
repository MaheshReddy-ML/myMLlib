import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression:
    """
    A simple implementation of Linear Regression using Gradient Descent.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate for gradient descent updates.
    epochs : int, default=1000
        Number of iterations for gradient descent.
    normalize : str or None, default="both"
        Controls normalization behavior:
        - "both" : normalize X and Y
        - "x"    : normalize only X
        - "y"    : normalize only Y
        - None   : no normalization applied

    Attributes
    ----------
    w : float
        Learned weight parameter (slope).
    b : float
        Learned bias parameter (intercept).
    x_mean, x_std : float
        Mean and standard deviation of X (if normalized).
    y_mean, y_std : float
        Mean and standard deviation of Y (if normalized).
    """

    def __init__(self, lr=0.01, epochs=1000, normalize="both"):
        self.lr = lr
        self.epochs = epochs
        self.normalize = normalize
        self.w = None
        self.b = None

    def _normalize_x(self, x):
        """Normalize feature values (X) using mean and standard deviation."""
        self.x_mean = np.mean(x)
        self.x_std = np.std(x) if np.std(x) != 0 else 1
        return (x - self.x_mean) / self.x_std

    def _normalize_y(self, y):
        """Normalize target values (Y) using mean and standard deviation."""
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) if np.std(y) != 0 else 1
        return (y - self.y_mean) / self.y_std

    def fit(self, x, y):
        """
        Train the linear regression model using gradient descent.

        Parameters
        ----------
        x : array-like
            Feature values.
        y : array-like
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        # Apply normalization based on user choice
        if self.normalize == "both":
            x = self._normalize_x(x)
            y = self._normalize_y(y)
        elif self.normalize == "x":
            x = self._normalize_x(x)
        elif self.normalize == "y":
            y = self._normalize_y(y)

        self.w = 0.0
        self.b = 0.0
        n = len(x)

        # Gradient Descent Loop
        for _ in range(self.epochs):
            y_pred = self.w * x + self.b
            error = y_pred - y
            dw = (2/n) * np.dot(x, error)
            db = (2/n) * np.sum(error)
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict(self, x):
        """
        Predict target values for given input features.

        Parameters
        ----------
        x : array-like
            Input feature values.

        Returns
        -------
        array
            Predicted target values.
        """
        x = np.array(x, dtype=float)

        # Apply normalization to X if needed
        if self.normalize in ["both", "x"]:
            x = (x - self.x_mean) / self.x_std

        y_pred = self.w * x + self.b

        # Denormalize Y if needed
        if self.normalize in ["both", "y"]:
            y_pred = y_pred * self.y_std + self.y_mean

        return y_pred

    def r2_score(self, y_true, y_pred):
        """
        Calculate the R² (coefficient of determination) score.

        Parameters
        ----------
        y_true : array-like
            Actual target values.
        y_pred : array-like
            Predicted target values.

        Returns
        -------
        float
            R² score.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def mse(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE).

        Parameters
        ----------
        y_true : array-like
            Actual target values.
        y_pred : array-like
            Predicted target values.

        Returns
        -------
        float
            Mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def plot_fit(self, x, y, xlabel="X", ylabel="Y"):
        """
        Plot the actual vs predicted values along with the regression line.

        Parameters
        ----------
        x : array-like
            Feature values.
        y : array-like
            Target values.
        xlabel : str, default="X"
            Label for the X-axis.
        ylabel : str, default="Y"
            Label for the Y-axis.
        """
        plt.scatter(x, y, color="green", label="Actual")
        plt.plot(x, self.predict(x), color="red", label="Predicted")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
