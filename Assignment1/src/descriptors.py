"""Descritores (SIFT, ORB, KAZE, BRIEF, BRISK, FREAK, BRIEF próprio) e extração completa.  [Pessoa A]"""

import time

import cv2
import numpy as np

from brief_own import STRATEGIES, compute_brief, make_pairs
from detectors import create_detector, detect
from interface import Features

# Distância usada no matching: L2 nos descritores float, Hamming nos binários (enunciado §2.5)
FLOAT_DESCRIPTORS = ["SIFT", "KAZE"]

# Nomes dos descritores BRIEF próprios: "BRIEF-GI", "BRIEF-GII", "BRIEF-GIII", "BRIEF-GV"
BRIEF_OWN_NAMES = ["BRIEF-" + s for s in STRATEGIES]


def descriptor_norm(name: str) -> int:
    return cv2.NORM_L2 if name in FLOAT_DESCRIPTORS else cv2.NORM_HAMMING


def create_descriptor(name: str, cfg: dict):
    """Cria o descritor do OpenCV com os parâmetros do config."""
    if name in ("SIFT", "ORB", "KAZE"):
        return create_detector(name, cfg)   # nestes métodos o mesmo objeto deteta e descreve
    params = cfg["descriptors"][name]
    if name == "BRIEF":
        return cv2.xfeatures2d.BriefDescriptorExtractor_create(**params)
    if name == "BRISK":
        return cv2.BRISK_create(**params)
    if name == "FREAK":
        return cv2.xfeatures2d.FREAK_create(**params)
    raise ValueError(f"Descritor desconhecido: {name}")


def describe(name: str, gray, kps, cfg: dict):
    """Calcula os descritores dos keypoints. Devolve (keypoints, descritores, tempo em ms).

    Atenção: os descritores removem keypoints (por exemplo, junto à borda), por isso a lista
    devolvida pode ser mais curta e é essa que tem de ser usada. O tempo não inclui a preparação
    (criar o objeto do OpenCV ou gerar os pares do BRIEF próprio), tal como na deteção (D06).
    """
    if name in BRIEF_OWN_NAMES:
        pairs = make_pairs(name.removeprefix("BRIEF-"), cfg)
        t0 = time.perf_counter()
        kps, desc = compute_brief(gray, kps, pairs, cfg)
    else:
        descriptor = create_descriptor(name, cfg)
        t0 = time.perf_counter()
        kps, desc = descriptor.compute(gray, list(kps))
    t_desc = (time.perf_counter() - t0) * 1000

    kps = list(kps) if kps is not None else []
    if desc is None or len(kps) == 0:
        return [], None, t_desc
    if name in FLOAT_DESCRIPTORS:
        desc = desc.astype(np.float32, copy=False)
    return kps, desc, t_desc


def split_method(method: str):
    """"FAST+BRIEF" -> ("FAST", "BRIEF"); "SIFT" -> ("SIFT", "SIFT")."""
    if "+" in method:
        detector, descriptor = method.split("+")
    else:
        detector = descriptor = method
    # SIFT, ORB e KAZE dependem da escala/octava do seu próprio detetor (D11)
    if descriptor in ("SIFT", "ORB", "KAZE") and descriptor != detector:
        raise ValueError(f"O descritor {descriptor} só funciona com o detetor {descriptor}")
    return detector, descriptor


def extract(method: str, gray, cfg: dict) -> Features:
    """Deteta e descreve com um método, por exemplo "SIFT" ou "FAST+FREAK".
    A deteção e a descrição são feitas em separado para medir os dois tempos (D06)."""
    detector, descriptor = split_method(method)
    kps, t_det = detect(detector, gray, cfg)
    n_detected = len(kps)
    kps, desc, t_desc = describe(descriptor, gray, kps, cfg)
    return Features(kps, desc, descriptor_norm(descriptor), t_det, t_desc, method, n_detected)
