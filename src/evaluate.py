"""Métricas para un problema de priorización de llamadas con clase positiva minoritaria."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def capture_at_k(y_true, scores, k=0.20):
    """Fracción de los suscriptores que quedan dentro del top-k% con mayor score."""
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(len(y_true) * k))
    top = np.argsort(-np.asarray(scores), kind="stable")[:n_top]
    return y_true[top].sum() / y_true.sum()


def ranking_metrics(y_true, scores, k=0.20):
    """Métricas que no dependen de un umbral."""
    captura = capture_at_k(y_true, scores, k)
    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        f"captura_top{int(k * 100)}": captura,
        f"lift_top{int(k * 100)}": captura / k,
    }


def threshold_metrics(y_true, scores, threshold):
    """Métricas de la decisión llamar / no llamar para un umbral dado."""
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "umbral": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "pct_llamados": pred.mean(),
    }


def gain_curve(y_true, scores):
    """Curva de ganancia acumulada: % de clientes llamados vs % de suscriptores captados."""
    y_true = np.asarray(y_true)
    orden = np.argsort(-np.asarray(scores), kind="stable")
    captados = np.cumsum(y_true[orden]) / y_true.sum()
    llamados = np.arange(1, len(y_true) + 1) / len(y_true)
    return pd.DataFrame({"pct_llamados": llamados, "pct_captados": captados})
