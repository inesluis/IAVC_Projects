# Formato comum do resultado de detetar + descrever uma imagem  

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
    raw: dict | None = None   # só nos métodos aprendidos: tensores do extrator, que o LightGlue recebe

def keypoints_to_array(kps) -> np.ndarray:
    # Coordenadas (x, y) dos keypoints como array float64 (N, 2) 
    return np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)

def descriptor_size(desc: np.ndarray | None) -> tuple[int, int, int]:
    # Devolve (dimensão, bytes por descritor, bytes no total).
    # A dimensão é o nº de bits nos descritores binários (uint8) e o nº de valores nos float.
    if desc is None:
        return 0, 0, 0
    is_binary = desc.dtype == np.uint8
    dim = desc.shape[1] * 8 if is_binary else desc.shape[1]
    bytes_per_kp = desc.shape[1] * desc.itemsize
    return dim, bytes_per_kp, desc.nbytes