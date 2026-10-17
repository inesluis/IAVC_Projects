# Formato comum do resultado de detetar + descrever uma imagem  

from dataclasses import dataclass
import numpy as np

@dataclass
class Features:
    kps: list                  
    desc: np.ndarray | None    
    norm: int                 
    t_det: float               
    t_desc: float              
    method: str                
    n_detected: int = 0        
    raw: dict | None = None    

def keypoints_to_array(kps) -> np.ndarray:
    # coordenadas (x, y) dos keypoints como array float64 (N, 2) 
    return np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)

def descriptor_size(desc: np.ndarray | None) -> tuple[int, int, int]:
    # devolve (dimensão, bytes por descritor, bytes no total).
    # a dimensão é o nº de bits nos descritores binários (uint8) e o nº de valores nos float.
    if desc is None:
        return 0, 0, 0
    is_binary = desc.dtype == np.uint8
    dim = desc.shape[1] * 8 if is_binary else desc.shape[1]
    bytes_per_kp = desc.shape[1] * desc.itemsize
    return dim, bytes_per_kp, desc.nbytes