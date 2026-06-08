"""
Shared preprocessing transformer used by both the model-save script
and app.py.  Must live in a stable importable module so pickle can
resolve the class when loading diabetes_model.pkl.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class RenameDPF(BaseEstimator, TransformerMixin):
    """
    Step 1 of the inference pipeline:
    - Renames 'DiabetesPedigreeFunction' → 'DPF'
    - Replaces zero values in clinical columns with the column mean
      (mirrors the notebook's preprocessing)
    """

    def __init__(self, zero_cols=None):
        self.zero_cols = zero_cols or ZERO_COLS

    def fit(self, X, y=None):
        # Compute means from training data for zero-imputation
        X = self._as_df(X)
        self.col_means_ = {}
        for col in self.zero_cols:
            if col in X.columns:
                tmp = X[col].replace(0, np.nan)
                self.col_means_[col] = tmp.mean()
        return self

    def transform(self, X):
        X = self._as_df(X).copy()
        if "DiabetesPedigreeFunction" in X.columns:
            X = X.rename(columns={"DiabetesPedigreeFunction": "DPF"})
        for col in self.zero_cols:
            if col in X.columns:
                mean_val = self.col_means_.get(col, X[col].mean())
                X[col] = X[col].replace(0, np.nan).fillna(mean_val)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(
            ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
             "Insulin", "BMI", "DPF", "Age"]
        )

    @staticmethod
    def _as_df(X):
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X)
        return X
