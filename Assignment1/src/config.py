"""Parâmetros únicos de todas as experiências (A + B): eps 3 px, ratio 0.8, max_keypoints,
FAST, BRIEF próprio, RANSAC, seeds e caminhos. Nenhum módulo deve ter parâmetros fixos no código.

Cada escolha está justificada em DECISOES.md (referência Dxx ao lado de cada parâmetro).
"""

import copy
from pathlib import Path

import cv2
import numpy as np

# Raiz do projeto (pasta Assignment1/), independente da pasta de onde se corre o script.
ROOT = Path(__file__).resolve().parents[1]

CFG = {
    "seed": 0,                       # D07: seeds fixas (numpy, OpenCV)
    "num_threads": None,             # D12: None = omissão do OpenCV; inteiro fixa nº de threads

    "paths": {
        "hpatches": ROOT / "data" / "hpatches-sequences-release",   # D04
        "graf": ROOT / "data" / "graf",
        "panorama": ROOT / "data" / "Panorama",
        "tables": ROOT / "results" / "tables",
        "figures": ROOT / "results" / "figures",
        "panoramas": ROOT / "results" / "panoramas",
    },

    # ---- Protocolo de avaliação (enunciado §2.8) ----
    "eval": {
        "eps_px": 3.0,                # correto se ||H_gt·p - p'|| < 3 px
        "pairs_k": [2, 3, 4, 5, 6],   # pares 1->k de cada sequência HPatches
        "categories": ["i", "v"],     # iluminação / ponto de vista
    },

    # ---- Orçamento de pontos comum a todos os métodos (D05) ----
    "max_keypoints": 2000,

    # ---- Detetores (D09, D10) ----
    "detectors": {
        "SIFT": {"nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
        "ORB": {"scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 31, "fastThreshold": 20},
        "KAZE": {"extended": False, "upright": False, "threshold": 0.001},
        "FAST": {"threshold": 20, "nonmaxSuppression": True, "type": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16},
    },

    # ---- Descritores (D11) ----
    "descriptors": {
        "BRIEF": {"bytes": 32, "use_orientation": False},
        "BRISK": {"thresh": 30, "octaves": 3, "patternScale": 1.0},
        "FREAK": {"orientationNormalized": True, "scaleNormalized": True,
                  "patternScale": 22.0, "nOctaves": 4},
    },

    # ---- Matching (enunciado §2.8) ----
    "matching": {
        "ratio": 0.8,          # rejeita se d1/d2 >= 0.8
        "cross_check": True,   # 1-para-1
    },

    # ---- RANSAC (Pessoa B; usado também pela avaliação, D03) ----
    "ransac": {"thr_px": 3.0, "max_iters": 2000, "confidence": 0.995},
}


def get_config(**overrides) -> dict:
    """Devolve uma cópia profunda de CFG, para cada experiência poder alterar parâmetros
    sem afetar as outras. Ex.: get_config(max_keypoints=None)."""
    cfg = copy.deepcopy(CFG)
    cfg.update(overrides)
    return cfg


def setup_env(cfg: dict) -> np.random.Generator:
    """Fixa seeds (OpenCV e NumPy) e nº de threads do OpenCV. Chamar uma vez no início
    de cada experiência. Devolve o gerador NumPy a usar (np.random.default_rng(seed))."""
    cv2.setRNGSeed(cfg["seed"])
    if cfg.get("num_threads") is not None:
        cv2.setNumThreads(int(cfg["num_threads"]))
    return np.random.default_rng(cfg["seed"])
