"""Farm detection model module.

This module provides a Gaussian Naive Bayes classifier with integrated preprocessing
and label encoding capabilities for farm detection tasks.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from farm_detection.data.preprocess import Preprocessor


class GNBWithEncoding(Preprocessor):
    """Gaussian Naive Bayes model with preprocessing and label encoding.

    Args:
        Preprocessor (Preprocessor): Base class for preprocessing.
    """

    def __init__(self, priors=None, var_smoothing=1e-9):
        """Initialize the Gaussian Naive Bayes model with optional priors and variance smoothing.

        Args:
            priors (list, optional): Prior probabilities of the classes. Defaults to None.
            var_smoothing (float, optional): Portion of the largest variance of all features added to variances for calculation stability. Defaults to 1e-9.
        """
        super().__init__()
        self.label_encoder = LabelEncoder()
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gnb", GaussianNB(priors=priors, var_smoothing=var_smoothing)),
            ]
        )

    def fit(self, X, y):
        """Fit the Gaussian Naive Bayes model to the training data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            GNBWithEncoding: Fitted model instance.
        """
        X = self.log_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)
        return self

    def predict(self, X):
        """Predict class labels for samples in X using the trained model.

        Args:
            X (pd.DataFrame): Feature matrix to predict on.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X = self.log_transform(X)
        y_pred = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
