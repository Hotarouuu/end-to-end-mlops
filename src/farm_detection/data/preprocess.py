from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import joblib


class Preprocessor:
    def __init__(self):
        pass

    def log_transform(self, X):
        X["humidity_log"] = np.log(X["humidity"] + 1)
        X["rainfall_log"] = np.log(X["rainfall"] + 1)
        X.drop(["humidity", "rainfall"], axis=1, inplace=True)
        return X
