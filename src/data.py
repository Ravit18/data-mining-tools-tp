"""Descarga, carga y separación del dataset Bank Marketing (UCI id=222)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def download_raw(path=config.DATA_RAW):
    """Descarga el dataset desde UCI y lo guarda tal cual en data/raw."""
    from ucimlrepo import fetch_ucirepo

    bank = fetch_ucirepo(id=config.UCI_ID)
    df = pd.concat([bank.data.features, bank.data.targets], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_raw(path=config.DATA_RAW):
    """Lee el CSV crudo; si no existe, lo descarga primero."""
    if not path.exists():
        return download_raw(path)
    return pd.read_csv(path)


def clean(df):
    """Correcciones que no dependen de los datos (no aprenden nada del train).

    - `day_of_week` en UCI contiene valores 1-31: es el día del mes, se renombra a `day`.
    - `y` pasa de yes/no a 1/0.
    - Se agrega `orden`, la posición original de la fila (el archivo viene ordenado por fecha).
    """
    df = df.rename(columns={"day_of_week": "day"}).copy()
    df[config.TARGET] = (df[config.TARGET] == "yes").astype(int)
    df["orden"] = range(len(df))
    return df


def split_xy(df, drop_leakage=True):
    """Separa X e y. Por defecto descarta las columnas con fuga de información."""
    drop = [config.TARGET, "orden"]
    if drop_leakage:
        drop += config.LEAKAGE_COLS
    return df.drop(columns=drop), df[config.TARGET]


def train_val_test_split(df, seed=config.SEED):
    """Split estratificado 60/20/20 sobre el DataFrame limpio."""
    resto, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=seed
    )
    val_rel = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        resto, test_size=val_rel, stratify=resto[config.TARGET], random_state=seed
    )
    return train, val, test


if __name__ == "__main__":
    raw = download_raw()
    print(f"Guardado {config.DATA_RAW} con {raw.shape[0]} filas y {raw.shape[1]} columnas")
