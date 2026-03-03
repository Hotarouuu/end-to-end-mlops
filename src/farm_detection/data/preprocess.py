from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class Preprocessor:
    def __init__(self, joblib_file=None):
        if joblib_file:
            self.scaler_path = joblib_file["scaler"]
            self.encoder_path = joblib_file["labelencoder"]
        else:
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()

    def log_transform(self, X):
        X["humidity_log"] = np.log(X["humidity"] + 1)
        X["rainfall_log"] = np.log(X["rainfall"] + 1)
        X.drop(["humidity", "rainfall"], axis=1, inplace=True)
        return X

    def fit_transform(self, X, y):
        X = self.log_transform(X)
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        return X_scaled, y_encoded

    def fit(self, X, y):
        X = self.log_transform(X)
        self.scaler.fit(X)
        self.label_encoder.fit(y)

    def transform(self, X, y):
        X = self.log_transform(X)
        X_scaled = self.scaler.transform(X)
        y_encoded = self.label_encoder.transform(y)
        return X_scaled, y_encoded
