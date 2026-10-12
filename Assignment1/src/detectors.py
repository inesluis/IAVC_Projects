"""Detetores de keypoints: SIFT, ORB, KAZE e FAST.  [Pessoa A]"""

import time

import cv2


def create_detector(name: str, cfg: dict):
    """Cria o detetor do OpenCV com os parâmetros de cfg["detectors"]."""
    params = cfg["detectors"][name]
    if name == "SIFT":
        return cv2.SIFT_create(**params)
    if name == "ORB":
        # o ORB precisa do limite na criação: distribui os pontos pelos níveis da pirâmide (D10)
        return cv2.ORB_create(nfeatures=cfg["max_keypoints"] or 100_000, **params)
    if name == "KAZE":
        return cv2.KAZE_create(**params)
    if name == "FAST":
        return cv2.FastFeatureDetector_create(**params)
    raise ValueError(f"Detetor desconhecido: {name}")


def keep_strongest(kps, n_max):
    """Fica com os n_max keypoints de maior resposta (D05). n_max = None fica com todos.
    Nota: o FAST sem NMS devolve resposta 0 em todos os pontos; com limite, usar sempre NMS (D24)."""
    kps = list(kps)
    if n_max is None or len(kps) <= n_max:
        return kps
    return sorted(kps, key=lambda k: -k.response)[:n_max]


def detect(name: str, gray, cfg: dict):
    """Deteta keypoints numa imagem em cinzento. Devolve (keypoints, tempo em ms).
    O tempo não inclui a criação do detetor (D06)."""
    detector = create_detector(name, cfg)
    t0 = time.perf_counter()
    kps = keep_strongest(detector.detect(gray, None), cfg["max_keypoints"])
    t_det = (time.perf_counter() - t0) * 1000
    return kps, t_det
