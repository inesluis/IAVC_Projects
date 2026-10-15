# Matching de descritores: ratio test de Lowe + cross-check 

import time
import cv2
import numpy as np
from interface import Features

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

def make_matcher(norm: int, kind: str, cfg: dict):
    """Pesquisa de vizinhos: "bf" (força bruta, exata) ou "flann" (vizinho aproximado).

    No FLANN, os descritores float usam KD-trees aleatórias e os binários usam LSH (tabelas de hash),
    porque as KD-trees só servem para a distância L2.
    """
    if kind == "bf":
        return cv2.BFMatcher(norm)   # Hamming nos binários, L2 nos float
    flann = cfg["matching"]["flann"]
    if norm == cv2.NORM_L2:
        index = {"algorithm": FLANN_INDEX_KDTREE, "trees": flann["kdtree_trees"]}
    else:
        index = {"algorithm": FLANN_INDEX_LSH, "table_number": flann["lsh_tables"],
                 "key_size": flann["lsh_key_size"], "multi_probe_level": flann["lsh_multi_probe"]}
    return cv2.FlannBasedMatcher(index, {"checks": flann["checks"]})

def match_descriptors(f1: Features, f2: Features, cfg: dict, matcher: str = "bf"):
    # Emparelha os descritores da imagem 1 com os da imagem 2. Devolve (matches, tempo em ms).
    """
    Um match é aceite se passar os dois filtros:
      1. ratio test: a distância ao melhor vizinho é menor que ratio x a distância ao segundo;
      2. cross-check: o melhor vizinho da imagem 2 para a imagem 1 é o mesmo ponto.
    O resultado é 1-para-1. Em cada cv2.DMatch, queryIdx é o índice na imagem 1 e trainIdx na imagem 2.
    matcher="flann" só muda a pesquisa de vizinhos (estudo força bruta vs aproximado); os filtros
    são os mesmos. Nas experiências usa-se sempre a força bruta.
    """
    if f1.desc is None or f2.desc is None:
        return [], 0.0
    ratio = cfg["matching"]["ratio"]

    t0 = time.perf_counter()
    matcher = make_matcher(f1.norm, matcher, cfg)
    two_best = matcher.knnMatch(f1.desc, f2.desc, k=2)
    back = matcher.knnMatch(f2.desc, f1.desc, k=1)

    # melhor vizinho de cada ponto da imagem 2 na imagem 1
    best_in_1 = {m[0].queryIdx: m[0].trainIdx for m in back if m}

    matches = []
    for pair in two_best:
        if len(pair) < 2:      # a imagem 2 só tem um ponto: o ratio não está definido
            continue
        best, second = pair
        if best.distance < ratio * second.distance and best_in_1.get(best.trainIdx) == best.queryIdx:
            matches.append(best)
    t_match = (time.perf_counter() - t0) * 1000
    return matches, t_match

def matched_points(f1: Features, f2: Features, matches):
    # Coordenadas dos pontos emparelhados: dois arrays (M, 2), um por imagem 
    p1 = np.array([f1.kps[m.queryIdx].pt for m in matches], dtype=np.float64).reshape(-1, 2)
    p2 = np.array([f2.kps[m.trainIdx].pt for m in matches], dtype=np.float64).reshape(-1, 2)
    return p1, p2