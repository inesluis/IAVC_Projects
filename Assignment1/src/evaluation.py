# Avaliação: métricas de um par de imagens, corrida sobre muitos pares e tabelas-resumo 

import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from homography import corner_distance, estimate_homography
from interface import descriptor_size, keypoints_to_array
from matching import matched_points
from methods import extract_features, match_features

# -------------------------------------------- Geometria

def project(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    # Aplica a homografia H a pontos (N, 2) 
    if len(points) == 0:
        return np.empty((0, 2))
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
 
def inside(points: np.ndarray, shape) -> np.ndarray:
    # Máscara dos pontos que caem dentro de uma imagem com forma (altura, largura) 
    h, w = shape[:2]
    return (points[:, 0] >= 0) & (points[:, 0] <= w - 1) & (points[:, 1] >= 0) & (points[:, 1] <= h - 1)
 
def match_errors(f1, fk, matches, H: np.ndarray) -> np.ndarray:
    # Erro de cada match em píxeis: distância entre H · p1 e p2 
    p1, p2 = matched_points(f1, fk, matches)
    return np.linalg.norm(project(p1, H) - p2, axis=1)
 
def gt_correspondences(p1: np.ndarray, pk: np.ndarray, H: np.ndarray, shape1, shapek, eps: float):
    # Conta as correspondências ground-truth entre os keypoints das duas imagens   
 
    if len(p1) == 0 or len(pk) == 0:
        return 0, 0, 0
    p1_in_k = project(p1, H)
    a = p1_in_k[inside(p1_in_k, shapek)]
    b = pk[inside(project(pk, np.linalg.inv(H)), shape1)]
    if len(a) == 0 or len(b) == 0:
        return 0, len(a), len(b)

    # pares de pontos a menos de eps: a KD-tree evita calcular todas as distâncias  
    near = cKDTree(a).sparse_distance_matrix(cKDTree(b), eps, output_type="ndarray")
    near = near[near["v"] < eps]
    if len(near) == 0:
        return 0, len(a), len(b)

    # emparelhamento 1-para-1  
    graph = csr_matrix((np.ones(len(near), dtype=np.int8), (near["i"], near["j"])), shape=(len(a), len(b)))
    matched = maximum_bipartite_matching(graph, perm_type="column")
    return int((matched >= 0).sum()), len(a), len(b)

# --------------------------------------------------- Métricas de um par

def evaluate_pair(f1, fk, matches, t_match: float, pair, cfg: dict) -> dict:
    # Métricas de um par de imagens, como uma linha da tabela de resultados.
    eps = cfg["eval"]["eps_px"]
    n_corresp, n_vis1, n_visk = gt_correspondences(
        keypoints_to_array(f1.kps), keypoints_to_array(fk.kps), pair.H, pair.gray1.shape, pair.grayk.shape, eps)

    n_features = len(f1.kps)
    n_putative = len(matches)
    n_correct = int((match_errors(f1, fk, matches, pair.H) < eps).sum())

    # homografia estimada com RANSAC a partir dos matches putativos (None se falhar)
    H_est, inliers, reproj_err = estimate_homography(f1.kps, fk.kps, matches, cfg)
    n_inliers = int(inliers.sum()) if H_est is not None else 0

    # tempos: média das duas imagens na deteção e na descrição; o matching é por par
    t_det = (f1.t_det + fk.t_det) / 2
    t_desc = (f1.t_desc + fk.t_desc) / 2
    desc_dim, desc_bytes_per_kp, desc_bytes = descriptor_size(f1.desc)

    return {
        "N_kps1": len(f1.kps), "N_kpsk": len(fk.kps), "N_detected1": f1.n_detected,
        "N_vis1": n_vis1, "N_visk": n_visk, "N_corresp": n_corresp,
        "repeatability": n_corresp / min(n_vis1, n_visk) if min(n_vis1, n_visk) > 0 else np.nan,
        "N_features": n_features, "N_putative": n_putative, "N_correct": n_correct,
        "PMR": n_putative / n_features if n_features else 0.0,
        "precision": n_correct / n_putative if n_putative else 0.0,
        "matching_score": n_correct / n_features if n_features else 0.0,
        "recall": n_correct / n_corresp if n_corresp else np.nan,
        "H_found": H_est is not None, "N_inliers": n_inliers,
        "inlier_ratio": n_inliers / n_putative if n_putative else 0.0,
        "reproj_err_px": reproj_err,
        "corner_err_px": corner_distance(H_est, pair.H, pair.gray1),   # exatidão da homografia
        "t_det_ms": t_det, "t_desc_ms": t_desc, "t_match_ms": t_match,
        "t_total_ms": 2 * (t_det + t_desc) + t_match,
        "desc_dim": desc_dim, "desc_bytes_per_kp": desc_bytes_per_kp, "desc_mem_kb": desc_bytes / 1024,
    }

# ---------------------------------------------------  Corrida sobre pares

def warm_up(methods, pair, cfg: dict) -> None:
    for method in methods:
        f1 = extract_features(method, pair.gray1, cfg)
        fk = extract_features(method, pair.grayk, cfg)
        match_features(method, f1, fk, cfg)

def evaluate_pairs(methods, pairs, cfg: dict) -> pd.DataFrame:
    rows = []
    current_seq = None
    features_1 = {}           
    t_start = time.perf_counter()

    for pair in pairs:
        if current_seq is None:
            warm_up(methods, pair, cfg) # 1) aquecimento 
        if pair.seq != current_seq: # 2) se nova sequência: recalcular features da imagem 1
            current_seq = pair.seq
            features_1 = {}
            print(f"{pair.seq}  ({time.perf_counter() - t_start:.0f} s)", flush=True)

        for method in methods: # 3) detetar, descrever e emparelhar cada método 
            if method not in features_1:
                features_1[method] = extract_features(method, pair.gray1, cfg)
            f1 = features_1[method]
            fk = extract_features(method, pair.grayk, cfg)
            matches, t_match = match_features(method, f1, fk, cfg)

            row = {"method": method, "seq": pair.seq, "category": pair.category, "k": pair.k,
                   "control": pair.control}
            row.update(evaluate_pair(f1, fk, matches, t_match, pair, cfg))
            rows.append(row)
    return pd.DataFrame(rows)

# -------------------------------------------------- Tabelas-Resumo

def homography_table(no_control: pd.DataFrame, cfg: dict) -> pd.DataFrame:
 
    by_category = no_control.groupby(["method", "category"])
    t4 = by_category[["N_putative", "N_inliers", "inlier_ratio", "reproj_err_px"]].mean()
    t4.insert(0, "n_pairs", by_category.size())
    t4["H_found"] = by_category["H_found"].mean()
    t4["corner_err_median_px"] = by_category["corner_err_px"].median()
    for thr in cfg["eval"]["corner_thr_px"]:
        t4[f"acc_{thr}px"] = (no_control["corner_err_px"] <= thr).groupby(
            [no_control["method"], no_control["category"]]).mean()
    return t4

def summarize(df: pd.DataFrame, cfg: dict) -> dict:
 
    no_control = df[~df["control"]]
    by_category = no_control.groupby(["method", "category"])

    t1 = by_category[["N_kps1", "N_vis1", "N_corresp", "repeatability"]].mean()
    t2 = by_category[["N_features", "N_putative", "N_correct", "N_corresp",
                      "PMR", "precision", "matching_score", "recall"]].mean()
    t2.insert(0, "n_pairs", by_category.size())
    t5 = no_control.groupby("method")[["t_det_ms", "t_desc_ms", "t_match_ms", "t_total_ms",
                                       "desc_dim", "desc_bytes_per_kp", "desc_mem_kb"]].mean()
    f1 = df.groupby(["method", "category", "k"])[["precision", "matching_score", "repeatability"]].mean()
    tables = {"T1_detection": t1, "T2_matching": t2, "T5_cost": t5, "F1_by_k": f1}
    if "corner_err_px" in df:    
        tables["T4_homography"] = homography_table(no_control, cfg)
    return tables

def robustness_table(tables: dict) -> pd.DataFrame:
    table = tables["T2_matching"][["PMR", "precision", "matching_score", "recall"]].join(
        tables["T1_detection"][["repeatability"]]).unstack("category")
    return table.reindex(columns=["i", "v", "rot", "scale"], level=1)

def save_tables(tables: dict, folder, prefix: str) -> None: 
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{prefix}{name}.csv", float_format="%.4f")