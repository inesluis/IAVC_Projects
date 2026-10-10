"""Detetores clássicos: SIFT, ORB, KAZE, FAST (com threshold e NMS configuráveis).  [Pessoa A]"""

import time

import cv2
import numpy as np

DETECTORS = ("SIFT", "ORB", "KAZE", "FAST")

# O ORB exige nfeatures > 0; usado só quando max_keypoints = None (sem orçamento). D10
_ORB_UNLIMITED = 100_000


def check_gray(gray: np.ndarray) -> None:
    """Garante que a entrada é uma imagem em cinzento uint8 (H, W)."""
    if gray is None or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Esperada imagem em cinzento uint8 (H, W)")


def create_detector(name: str, cfg: dict):
    """Cria o objeto OpenCV do detetor `name` com os parâmetros de cfg["detectors"][name]."""
    p = cfg["detectors"].get(name, {})
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=0, **p)      # o corte ao orçamento é feito em cap_keypoints
    if name == "ORB":
        return cv2.ORB_create(nfeatures=cfg.get("max_keypoints") or _ORB_UNLIMITED, **p)
    if name == "KAZE":
        return cv2.KAZE_create(**p)
    if name == "FAST":
        return cv2.FastFeatureDetector_create(**p)
    raise ValueError(f"Detetor desconhecido: {name} (disponíveis: {DETECTORS})")


def cap_keypoints(kps, n_max: int | None) -> list[cv2.KeyPoint]:
    """Mantém os n_max keypoints com maior resposta (D05). Ordenação estável -> determinística.
    n_max = None mantém todos.
    ATENÇÃO (D24): o FAST com NMS desligada devolve response = 0 em todos os pontos; aí o corte guardaria
    só os primeiros pela ordem de varrimento (linhas de cima). Usar o FAST com NMS ligada quando há limite."""
    kps = list(kps)
    if n_max is None or len(kps) <= n_max:
        return kps
    return sorted(kps, key=lambda k: -k.response)[:n_max]


def detect(name: str, gray: np.ndarray, cfg: dict) -> tuple[list[cv2.KeyPoint], float]:
    """Deteta keypoints com o detetor `name` ("SIFT", "ORB", "KAZE", "FAST").

    Aplica o orçamento cfg["max_keypoints"] (maior resposta primeiro).
    Devolve (kps, t_det_ms). O tempo inclui a deteção e o corte, mas não a criação do objeto (D06).
    """
    check_gray(gray)
    det = create_detector(name, cfg)
    t0 = time.perf_counter()
    kps = cap_keypoints(det.detect(gray, None), cfg.get("max_keypoints"))
    t_det = (time.perf_counter() - t0) * 1e3
    return kps, t_det
