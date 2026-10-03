"""Preprocesamiento reproducible: variables derivadas + ColumnTransformer."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_features(X):
    """Transformaciones fila a fila (sin parámetros aprendidos, no generan leakage).

    `pdays == -1` es un código para "nunca contactado", no una cantidad de días.
    Se separa en un indicador y se deja pdays en 0 para esos clientes.
    """
    X = X.copy()
    X["contactado_antes"] = (X["pdays"] != -1).astype(int)
    X["pdays"] = X["pdays"].where(X["pdays"] != -1, 0)
    return X


def signed_log(x):
    """log1p que conserva el signo; `balance` tiene saldos negativos."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(extra_numeric=()):
    """ColumnTransformer que se ajusta únicamente con los datos de entrenamiento.

    `extra_numeric` permite añadir columnas numéricas sin transformar; solo se usa
    para el experimento que mide cuánto infla las métricas la variable `duration`.
    """
    num_scale = StandardScaler()
    num_signed_log = Pipeline([
        ("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num_log = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    # Los faltantes en categóricas son informativos (dependen del periodo de la campaña),
    # por eso se conservan como categoría propia en lugar de imputar la moda.
    cat_nominal = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cat_binary = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        [
            ("num", num_scale, config.NUM_SCALE),
            ("num_signed_log", num_signed_log, config.NUM_SIGNED_LOG),
            ("num_log", num_log, config.NUM_LOG),
            ("flags", "passthrough", config.NUM_FLAGS),
            ("cat", cat_nominal, config.CAT_NOMINAL),
            ("bin", cat_binary, config.CAT_BINARY),
            ("extra", "passthrough", list(extra_numeric)),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model, extra_numeric=()):
    """Pipeline completo: derivadas -> preprocesamiento -> modelo."""
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", build_preprocessor(extra_numeric)),
        ("model", model),
    ])
