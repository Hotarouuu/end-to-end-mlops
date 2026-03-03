from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib

class Preprocessor:
    def __init__(self, joblib_file=None):
        if joblib_file:
            processor = joblib.load(joblib_file)
            self.scaler = processor["scaler"]
            self.label_encoder = processor["labelencoder"]
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

    def transform(self, X, y, df=False):
        X = self.log_transform(X)
        X_scaled = self.scaler.transform(X)
        y_encoded = self.label_encoder.transform(y)

        if df:
            df_return = pd.DataFrame(X_scaled, columns=self.scaler.get_feature_names_out())
            df_return['label_encoded'] = y_encoded
            return df_return
        else:
            return X_scaled, y_encoded
