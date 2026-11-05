"""Métricas de avaliação para classificação e retrieval de imagens.

Usadas em Parte 1 (BoVW), Parte 2 (VLAD/FV) e Parte 5 (retrieval).
"""
import numpy as np
from sklearn.metrics import average_precision_score


def compute_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average Precision binária (+1 / -1 labels)."""
    return float(average_precision_score(labels, scores))


def compute_map(labels_list, scores_list) -> tuple:
    """Mean AP sobre múltiplas categorias.

    Returns:
        (mAP, [AP_cat1, AP_cat2, ...])
    """
    aps = [compute_ap(l, s) for l, s in zip(labels_list, scores_list)]
    return float(np.mean(aps)), aps


def compute_pk(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precisão no rank k — fração de positivos nos top-k resultados."""
    order = np.argsort(scores)[::-1]
    return float(np.mean(labels[order[:k]] > 0))


def compute_rk(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Recall no rank k — fração dos positivos recuperados nos top-k."""
    order = np.argsort(scores)[::-1]
    n_pos = int(np.sum(labels > 0))
    if n_pos == 0:
        return 0.0
    return float(np.sum(labels[order[:k]] > 0) / n_pos)
