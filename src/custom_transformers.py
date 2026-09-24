from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from config import NULOS_PIPELINE

class NullNumericImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores nulos em colunas numéricas conforme o modelo do config.
    Se necessário, adiciona flag de nulo.
    """
    def __init__(self, columns, modelo):
        self.columns = columns
        self.modelo = modelo
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        config = NULOS_PIPELINE[self.modelo.lower()]["numeric"]
        self.impute_value_ = config["impute"]
        self.add_flag_ = config["flag"]
        
        # Prepara os nomes das features de saída
        output_features = list(self.columns)
        if self.add_flag_:
            output_features.extend([f"{col}_isnull" for col in self.columns])
        self.feature_names_out_ = output_features
        
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.columns:
            if self.add_flag_:
                X_[f"{col}_isnull"] = X_[col].isnull().astype(int)
            X_[col] = X_[col].fillna(self.impute_value_)
        return X_[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class NullCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores nulos em colunas categóricas conforme o modelo do config.
    Se necessário, adiciona flag de nulo.
    """
    def __init__(self, columns, modelo):
        self.columns = columns
        self.modelo = modelo
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        config = NULOS_PIPELINE[self.modelo.lower()]["categorical"]
        self.impute_value_ = config["impute"]
        self.add_flag_ = config["flag"]
        
        # Prepara os nomes das features de saída
        output_features = list(self.columns)
        if self.add_flag_:
            output_features.extend([f"{col}_isnull" for col in self.columns])
        self.feature_names_out_ = output_features
        
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.columns:
            if self.add_flag_:
                X_[f"{col}_isnull"] = X_[col].isnull().astype(int)
            X_[col] = X_[col].fillna(self.impute_value_)
        return X_[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_