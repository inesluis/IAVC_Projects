"""Contrato partilhado A <-> B: dataclasses Features, MatchResult e Pipeline.
Alterar só com acordo dos dois elementos do grupo.

Regras (D01):
  - kps em coordenadas da imagem ORIGINAL (px); métodos que redimensionam reescalam antes de devolver.
  - desc alinhado com kps: float -> float32 (N, D); binário -> uint8 empacotado (N, D_bits/8).
  - matches: queryIdx indexa a imagem 1, trainIdx indexa a imagem 2; 1-para-1; putativos (antes do RANSAC).
  - tempos em ms, medidos com time.perf_counter().
"""

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


@dataclass
class Features:
    """Resultado de detetar + descrever UMA imagem.

    Atributos:
        kps:    lista FINAL de cv2.KeyPoint (depois de compute() remover pontos da borda).
        desc:   np.ndarray (N, D) alinhado com kps, ou None (só detetor, ou 0 keypoints).
        norm:   cv2.NORM_L2, cv2.NORM_HAMMING ou None (descritor não usado diretamente).
        t_det:  tempo de deteção em ms.
        t_desc: tempo de descrição em ms. Métodos conjuntos que não separam os tempos:
                t_det = total, t_desc = 0.0 e meta["joint_timing"] = True.
        method: nome legível (ex.: "FAST+BRIEF", "SuperPoint").
        meta:   extra específico do método (parâmetros, tensores p/ LightGlue, ...). A avaliação ignora-o.
    """
    kps: list[cv2.KeyPoint]
    desc: np.ndarray | None
    norm: int | None
    t_det: float
    t_desc: float
    method: str
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Número de keypoints."""
        return len(self.kps)

    @property
    def is_binary(self) -> bool:
        """True se o descritor é binário empacotado (uint8)."""
        return self.desc is not None and self.desc.dtype == np.uint8

    @property
    def desc_dim(self) -> int:
        """Dimensão do descritor: nº de bits (binário) ou nº de elementos (float). 0 se não há."""
        if self.desc is None or self.desc.ndim != 2:
            return 0
        return self.desc.shape[1] * 8 if self.is_binary else self.desc.shape[1]

    @property
    def desc_bytes_per_kp(self) -> int:
        """Bytes ocupados por um descritor (ex.: SIFT 128x4 = 512, ORB 32)."""
        if self.desc is None or self.desc.ndim != 2:
            return 0
        return self.desc.shape[1] * self.desc.itemsize

    @property
    def desc_bytes(self) -> int:
        """Memória total dos descritores da imagem, em bytes."""
        return 0 if self.desc is None else int(self.desc.nbytes)

    def points(self) -> np.ndarray:
        """Coordenadas dos keypoints como array float64 (N, 2)."""
        return np.array([k.pt for k in self.kps], dtype=np.float64).reshape(-1, 2)


@dataclass
class MatchResult:
    """Resultado de emparelhar img1 -> img2.

    Atributos:
        matches: lista de cv2.DMatch 1-para-1 (putativos, antes do RANSAC).
        t_match: tempo de matching em ms.
        method:  nome do matcher (ex.: "BF-ratio0.8-xcheck", "MNN", "LightGlue").
        meta:    extra (ex.: scores do LightGlue, nº de camadas usadas).
    """
    matches: list[cv2.DMatch]
    t_match: float
    method: str
    meta: dict = field(default_factory=dict)


class Extractor(Protocol):
    def __call__(self, gray: np.ndarray, cfg: dict) -> Features:
        """gray: uint8 (H, W). Devolve Features. Não lê o disco; parâmetros vêm só de cfg."""
        ...


class Matcher(Protocol):
    def __call__(self, f1: Features, f2: Features, cfg: dict) -> MatchResult:
        """Emparelha f1 -> f2. Tem de devolver matches 1-para-1."""
        ...


@dataclass
class Pipeline:
    """Configuração completa avaliável = extrator + matcher (separa detetor/descritor do matcher)."""
    name: str
    extract: Extractor
    match: Matcher
