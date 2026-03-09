"""Data preprocessing module for farm detection.

Provides utilities for transforming and preparing farm detection dataset features,
including log transformations and data normalization.
"""

import numpy as np


class Preprocessor:
    """Data preprocessing utility for farm detection dataset.

    Handles various data transformation operations including scaling,
    encoding, and log transformations for numerical features.
    """

    def __init__(self):
        pass

    def log_transform(self, X):
        """Apply log transformation to humidity and rainfall features.

        Applies natural log transformation with offset to humidity and rainfall
        columns, then drops the original columns.

        Args:
            X (pd.DataFrame): Input dataframe containing humidity and rainfall columns.

        Returns:
            pd.DataFrame: Transformed dataframe with humidity_log and rainfall_log columns.
        """
        X["humidity_log"] = np.log(X["humidity"] + 1)
        X["rainfall_log"] = np.log(X["rainfall"] + 1)
        X.drop(["humidity", "rainfall"], axis=1, inplace=True)
        return X
