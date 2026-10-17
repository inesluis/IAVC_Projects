# Matching de descritores: ratio test de Lowe + cross-check 

import time
import cv2
import numpy as np
from interface import Features

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

def make_matcher(norm: int, kind: str, cfg: dict):
    if kind == "bf":
        return cv2.BFMatcher(norm)    
    flann = cfg["matching"]["flann"]
    if norm == cv2.NORM_L2:
        index = {"algorithm": FLANN_INDEX_KDTREE, "trees": flann["kdtree_trees"]}
    else:
        index = {"algorithm": FLANN_INDEX_LSH, "table_number": flann["lsh_tables"],
                 "key_size": flann["lsh_key_size"], "multi_probe_level": flann["lsh_multi_probe"]}
    return cv2.FlannBasedMatcher(index, {"checks": flann["checks"]})

def match_descriptors(f1: Features, f2: Features, cfg: dict, matcher: str = "bf"):
    # emparelha os descritores da imagem 1 com os da imagem 2. Devolve (matches, tempo em ms) 
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
        if len(pair) < 2:       
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