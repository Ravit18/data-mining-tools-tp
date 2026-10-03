"""Rutas, semilla y listas de columnas compartidas por notebooks y scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "bank_marketing.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

UCI_ID = 222
SEED = 42
TARGET = "y"

# `duration` solo se conoce cuando la llamada ya terminó (y con ella el resultado),
# así que no puede usarse para decidir a quién llamar.
LEAKAGE_COLS = ["duration"]

# Proporciones del split estratificado train / validation / test
TEST_SIZE = 0.20
VAL_SIZE = 0.20

NUM_SCALE = ["age", "day"]
NUM_SIGNED_LOG = ["balance"]
NUM_LOG = ["campaign", "previous", "pdays"]
NUM_FLAGS = ["contactado_antes"]
CAT_NOMINAL = ["job", "marital", "education", "contact", "month", "poutcome"]
CAT_BINARY = ["default", "housing", "loan"]

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
