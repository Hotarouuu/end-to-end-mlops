"""Farm detection model module.

This module provides an XGBoost classifier with integrated preprocessing
and label encoding capabilities for farm detection tasks.
"""

from xgboost import XGBClassifier


class XGBWithEncoding:
    """XGBoost model with preprocessing and label encoding.

    Args:
        params (dict): Parameters passed to the XGBClassifier.
    """

    def __init__(self, params: dict):
        """Initialize the XGBoost model with provided parameters.

        Args:
            params (dict): Parameters passed to the XGBClassifier.
        """
        self.model = XGBClassifier(**params)

    def fit(self, X, y):
        """Fit the XGBoost model to the training data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            XGBWithEncoding: Fitted model instance.
        """
        y_encoded = y.astype("category").cat.codes
        self.model.fit(X, y_encoded)
        return self

    def predict(self, X):
        """Predict class labels for samples in X using the trained model.

        Args:
            X (pd.DataFrame): Feature matrix to predict on.

        Returns:
            np.ndarray: Predicted class labels.
        """
        y_pred = self.model.predict(X)
        return y_pred
