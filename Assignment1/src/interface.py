"""Formato comum do resultado de detetar + descrever uma imagem (combinado entre A e B, D01).

Regras:
  - kps em coordenadas da imagem original (píxeis);
  - desc alinhado com kps: float32 (N, D) nos descritores float; uint8 (N, bits/8) nos binários;
  - tempos em milissegundos.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Features:
    kps: list                 # lista de cv2.KeyPoint (depois de o descritor remover pontos da borda)
    desc: np.ndarray | None   # None se não sobrar nenhum keypoint
    norm: int                 # cv2.NORM_L2 ou cv2.NORM_HAMMING
    t_det: float              # tempo de deteção (ms)
    t_desc: float             # tempo de descrição (ms)
    method: str               # ex.: "SIFT", "FAST+BRIEF"
    n_detected: int = 0       # nº de keypoints antes de o descritor remover os da borda


def keypoints_to_array(kps) -> np.ndarray:
    """Coordenadas (x, y) dos keypoints como array float64 (N, 2)."""
    return np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)


def descriptor_size(desc: np.ndarray | None) -> tuple[int, int, int]:
    """Devolve (dimensão, bytes por descritor, bytes no total).
    A dimensão é o nº de bits nos descritores binários (uint8) e o nº de valores nos float."""
    if desc is None:
        return 0, 0, 0
    is_binary = desc.dtype == np.uint8
    dim = desc.shape[1] * 8 if is_binary else desc.shape[1]
    bytes_per_kp = desc.shape[1] * desc.itemsize
    return dim, bytes_per_kp, desc.nbytes
