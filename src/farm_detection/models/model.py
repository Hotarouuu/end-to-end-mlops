from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


class GNBWithEncoding:
    def __init__(self, priors=None, var_smoothing=1e-9):
        self.label_encoder = LabelEncoder()
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('gnb', GaussianNB(priors=priors, var_smoothing=var_smoothing))
        ])

    def log_transform(self, X):
        X["humidity_log"] = np.log(X["humidity"] + 1)
        X["rainfall_log"] = np.log(X["rainfall"] + 1)
        X.drop(["humidity", "rainfall"], axis=1, inplace=True)
        return X
    
    def fit(self, X, y):
        X = self.log_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        self.pipeline.fit(X, y_encoded)
        return self
    
    def predict(self, X):
        X = self.log_transform(X)
        y_pred = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
