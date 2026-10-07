"""Descritores clássicos: SIFT, ORB, KAZE, BRIEF, BRISK, FREAK; extract_classic(detetor+descritor).  [Pessoa A]"""

import functools
import time

import cv2
import numpy as np

from detectors import check_gray, create_detector, detect
from interface import Features

# Norma de matching de cada descritor: float -> L2, binário -> Hamming (enunciado §2.5).
DESCRIPTOR_NORM = {
    "SIFT": cv2.NORM_L2,
    "KAZE": cv2.NORM_L2,
    "ORB": cv2.NORM_HAMMING,
    "BRIEF": cv2.NORM_HAMMING,
    "BRISK": cv2.NORM_HAMMING,
    "FREAK": cv2.NORM_HAMMING,
}

# Descritores que só podem ser usados com o seu próprio detetor (dependem de escala/octava
# atribuídas por esse detetor). Os restantes funcionam sobre keypoints de qualquer detetor. D11
_OWN_DETECTOR_ONLY = {"SIFT", "KAZE", "ORB"}

# Configurações clássicas mínimas do enunciado (§4).
CLASSIC_COMBOS = ("SIFT", "ORB", "KAZE", "FAST+BRIEF", "FAST+BRISK", "FAST+FREAK")


def create_descriptor(name: str, cfg: dict):
    """Cria o objeto OpenCV que calcula o descritor `name`."""
    if name in ("SIFT", "ORB", "KAZE"):
        return create_detector(name, cfg)   # mesmo objeto/parâmetros que o detetor
    p = cfg["descriptors"].get(name, {})
    if name == "BRIEF":
        return cv2.xfeatures2d.BriefDescriptorExtractor_create(**p)
    if name == "BRISK":
        return cv2.BRISK_create(**p)
    if name == "FREAK":
        return cv2.xfeatures2d.FREAK_create(**p)
    raise ValueError(f"Descritor desconhecido: {name} (disponíveis: {tuple(DESCRIPTOR_NORM)})")


def describe(name: str, gray: np.ndarray, kps: list[cv2.KeyPoint], cfg: dict
             ) -> tuple[list[cv2.KeyPoint], np.ndarray | None, int, float]:
    """Calcula descritores `name` para os keypoints dados.

    Devolve (kps_filtrados, desc, norm, t_desc_ms). ATENÇÃO: compute() remove keypoints
    (ex.: perto da borda), por isso a lista devolvida pode ser mais curta e é a que deve ser usada.
    desc é None se não sobrar nenhum keypoint.
    """
    check_gray(gray)
    ext = create_descriptor(name, cfg)
    t0 = time.perf_counter()
    kps_out, desc = ext.compute(gray, list(kps))
    t_desc = (time.perf_counter() - t0) * 1e3
    kps_out = list(kps_out) if kps_out is not None else []
    if desc is None or len(kps_out) == 0:
        return [], None, DESCRIPTOR_NORM[name], t_desc
    if DESCRIPTOR_NORM[name] == cv2.NORM_L2:
        desc = desc.astype(np.float32, copy=False)
    return kps_out, desc, DESCRIPTOR_NORM[name], t_desc


def parse_combo(combo: str) -> tuple[str, str]:
    """"FAST+BRIEF" -> ("FAST", "BRIEF"); "SIFT" -> ("SIFT", "SIFT"). Valida a combinação."""
    det, _, desc = combo.partition("+")
    desc = desc or det
    if desc in _OWN_DETECTOR_ONLY and desc != det:
        raise ValueError(f"O descritor {desc} só pode ser usado com o detetor {desc} (pedido: {combo})")
    return det, desc


def extract_classic(combo: str, gray: np.ndarray, cfg: dict) -> Features:
    """Deteta + descreve com uma configuração clássica (ex.: "SIFT", "FAST+FREAK").

    Deteção e descrição são feitas em passos separados para medir os dois tempos (D06).
    meta["n_detected"] guarda o nº de pontos antes de compute() filtrar.
    """
    det_name, desc_name = parse_combo(combo)
    kps, t_det = detect(det_name, gray, cfg)
    n_detected = len(kps)
    kps, desc, norm, t_desc = describe(desc_name, gray, kps, cfg)
    return Features(kps=kps, desc=desc, norm=norm, t_det=t_det, t_desc=t_desc, method=combo,
                    meta={"detector": det_name, "descriptor": desc_name, "n_detected": n_detected})


def make_extractor(combo: str):
    """Devolve um Extractor (gray, cfg) -> Features para registar em methods.PIPELINES."""
    parse_combo(combo)   # falha cedo se a combinação for inválida
    return functools.partial(extract_classic, combo)
