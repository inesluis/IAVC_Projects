"""Matching de descritores: ratio test + cross-check, vizinho mais próximo mútuo, (FLANN).  [Pessoa A]

Todos os matchers devolvem matches 1-para-1 (queryIdx -> imagem 1, trainIdx -> imagem 2) num MatchResult.
"""

import time

import cv2
import numpy as np

from interface import Features, MatchResult


def _has_desc(f: Features) -> bool:
    return f.desc is not None and len(f.desc) > 0


def _create_matcher(norm: int, binary: bool, cfg: dict):
    """BFMatcher (exato) ou FlannBasedMatcher (aproximado) para a norma dada (D16).
    FLANN: KD-trees para descritores float, LSH para binários."""
    if cfg["matching"].get("backend", "bf") == "bf":
        return cv2.BFMatcher(norm, crossCheck=False)
    search = dict(checks=cfg["matching"].get("flann_checks", 50))
    if binary:
        index = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)   # FLANN_INDEX_LSH
    else:
        index = dict(algorithm=1, trees=5)                                            # FLANN_INDEX_KDTREE
    return cv2.FlannBasedMatcher(index, search)


def _backend_name(cfg: dict) -> str:
    return cfg["matching"].get("backend", "bf").upper()


def match_descriptors(f1: Features, f2: Features, cfg: dict) -> MatchResult:
    """Ratio test de Lowe + cross-check (D16). Implementa o Matcher; serve float e binário.

    1) para cada descritor de img1, os 2 vizinhos mais próximos em img2;
    2) aceita se d1 < ratio * d2 (cfg["matching"]["ratio"], 0.8);
    3) cross-check: aceita só se o melhor vizinho de img2 -> img1 for o mesmo ponto (1-para-1).
    O tempo inclui as duas pesquisas (1->2 e 2->1).
    """
    ratio = cfg["matching"]["ratio"]
    cross = cfg["matching"].get("cross_check", True)
    name = f"{_backend_name(cfg)}-ratio{ratio}" + ("-xcheck" if cross else "")
    if not (_has_desc(f1) and _has_desc(f2)):
        return MatchResult([], 0.0, name)
    if f1.norm != f2.norm:
        raise ValueError(f"Normas diferentes: {f1.method} vs {f2.method}")

    t0 = time.perf_counter()
    m = _create_matcher(f1.norm, f1.is_binary, cfg)
    knn12 = m.knnMatch(f1.desc, f2.desc, k=2)
    best21 = {}
    if cross:
        for pair in m.knnMatch(f2.desc, f1.desc, k=1):
            if pair:
                best21[pair[0].queryIdx] = pair[0].trainIdx
    good = []
    for pair in knn12:
        if len(pair) < 2:          # img2 com 1 só ponto, ou LSH sem 2 vizinhos: ratio indefinido
            continue
        a, b = pair
        if a.distance < ratio * b.distance and (not cross or best21.get(a.trainIdx) == a.queryIdx):
            good.append(a)
    t_match = (time.perf_counter() - t0) * 1e3
    return MatchResult(good, t_match, name, meta={"ratio": ratio, "cross_check": cross})


def mutual_nn(f1: Features, f2: Features, cfg: dict) -> MatchResult:
    """Só vizinho mais próximo mútuo, sem ratio test. Alternativa para descritores aprendidos
    (ex.: SuperPoint+MNN), onde o ratio 0.8 pode ser demasiado restritivo."""
    name = f"{_backend_name(cfg)}-MNN"
    if not (_has_desc(f1) and _has_desc(f2)):
        return MatchResult([], 0.0, name)
    t0 = time.perf_counter()
    m = _create_matcher(f1.norm, f1.is_binary, cfg)
    nn12 = [p[0] for p in m.knnMatch(f1.desc, f2.desc, k=1) if p]
    best21 = {p[0].queryIdx: p[0].trainIdx for p in m.knnMatch(f2.desc, f1.desc, k=1) if p}
    good = [a for a in nn12 if best21.get(a.trainIdx) == a.queryIdx]
    t_match = (time.perf_counter() - t0) * 1e3
    return MatchResult(good, t_match, name)


def matches_to_points(f1: Features, f2: Features, matches) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas (float64, (M, 2)) dos pontos emparelhados em img1 e img2."""
    p1 = np.array([f1.kps[m.queryIdx].pt for m in matches], dtype=np.float64).reshape(-1, 2)
    p2 = np.array([f2.kps[m.trainIdx].pt for m in matches], dtype=np.float64).reshape(-1, 2)
    return p1, p2
