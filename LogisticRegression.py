import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Logistic Regression implementation from scratch using Gradient Descent.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate for gradient descent updates.
    epochs : int, default=1000
        Number of iterations for gradient descent.
    normalize : {"x", "both", None}, default="x"
        Controls normalization behavior:
        - "x"    : normalize only features X
        - "both" : normalize both X and y (rare for classification)
        - None   : no normalization applied

    Attributes
    ----------
    w : float
        Learned weight parameter (slope).
    b : float
        Learned bias parameter (intercept).
    x_mean, x_std : float
        Mean and standard deviation of X (if normalized).
    loss_history : list
        Stores the loss value at each epoch during training.
    """

    def __init__(self, lr=0.01, epochs=1000, normalize="x"):
        self.lr = lr
        self.epochs = epochs
        self.normalize = normalize
        self.w = None
        self.b = None
        self.loss_history = []

    def _normalize(self, x):
        """Normalize features using mean and standard deviation."""
        self.x_mean = np.mean(x)
        self.x_std = np.std(x) if np.std(x) != 0 else 1
        return (x - self.x_mean) / self.x_std

    def _sigmoid(self, z):
        """Apply the sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def _bce_loss(self, y, y_pred):
        """
        Compute Binary Cross-Entropy (Log Loss).

        Parameters
        ----------
        y : array-like
            True binary labels (0 or 1).
        y_pred : array-like
            Predicted probabilities.

        Returns
        -------
        float
            Binary cross-entropy loss.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(1 / len(y)) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, x, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters
        ----------
        x : array-like
            Input feature values.
        y : array-like
            True binary labels (0 or 1).

        Returns
        -------
        self : object
            Trained LogisticRegression instance.
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        if self.normalize in ["x", "both"]:
            x = self._normalize(x)

        self.w = 0.0
        self.b = 0.0
        n = len(x)

        for _ in range(self.epochs):
            linear_output = self.w * x + self.b
            y_pred = self._sigmoid(linear_output)
            error = y_pred - y  # Correct sign for gradient

            dw = (1 / n) * np.dot(x, error)
            db = (1 / n) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = self._bce_loss(y, y_pred)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, x):
        """
        Predict probabilities for the positive class.

        Parameters
        ----------
        x : array-like
            Input feature values.

        Returns
        -------
        array
            Predicted probabilities for each input sample.
        """
        x = np.array(x, dtype=float)
        if self.normalize in ["x", "both"]:
            x = (x - self.x_mean) / self.x_std
        return self._sigmoid(self.w * x + self.b)

    def predict(self, x, threshold=0.5):
        """
        Predict binary class labels.

        Parameters
        ----------
        x : array-like
            Input feature values.
        threshold : float, default=0.5
            Probability threshold for classifying as 1.

        Returns
        -------
        array
            Predicted binary labels (0 or 1).
        """
        return (self.predict_proba(x) >= threshold).astype(int)

    def plot_loss(self):
        """Plot the loss curve over training epochs."""
        plt.plot(self.loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Convergence")
        plt.show()
