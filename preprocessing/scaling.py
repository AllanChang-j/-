from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    def __init__(self, scale: bool = True):
        self.scale = scale
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if scale else None
        self.feature_columns: list[str] = []

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> "TabularPreprocessor":
        self.feature_columns = list(feature_columns)
        matrix = df[self.feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        imputed = self.imputer.fit_transform(matrix)
        if self.scaler is not None:
            self.scaler.fit(imputed)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = df.copy()
        matrix = transformed[self.feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        imputed = self.imputer.transform(matrix)
        if self.scaler is not None:
            imputed = self.scaler.transform(imputed)
        processed = pd.DataFrame(imputed, columns=self.feature_columns, index=transformed.index, dtype=float)
        for column in self.feature_columns:
            transformed[column] = processed[column]
        return transformed

    def fit_transform(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        return self.fit(df, feature_columns).transform(df)
