"""Métricas (PMR, precision, matching score, recall, repetibilidade, corner error, tempos, memória)
e corrida no HPatches completo, separada por i_* / v_*.  [Pessoa A]

Protocolo (enunciado §2.8, DECISOES D02, D17, D18):
  - match correto  <=>  ||H_gt · p1 - p2|| < eps (3 px);
  - PMR = N_putative / N_features;  Precision = N_correct / N_putative;
    Matching score = N_correct / N_features;  Recall = N_correct / N_corresp.

# PENDENTE (2026-10-08): métricas da homografia estimada (inliers RANSAC, rácio de inliers, erro médio
# de reprojeção, corner error e % de pares com erro <= 1/3/5 px, tabela T4). Dependem de
# homography.estimate_homography (Pessoa B), ainda não implementado. Ver DECISOES.md, P05.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from datasets import iter_hpatches_pairs, list_sequences
from interface import Features, MatchResult, Pipeline
from matching import matches_to_points


# ---------------------------------------------------------------- geometria

def project(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Aplica a homografia H a pontos (N, 2). Devolve (N, 2) float64."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.empty((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def inside(pts: np.ndarray, shape) -> np.ndarray:
    """Máscara dos pontos (N, 2) dentro de uma imagem com forma `shape` (H, W[, C])."""
    h, w = shape[:2]
    return (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)


def match_errors(f1: Features, f2: Features, matches, H: np.ndarray) -> np.ndarray:
    """Erro de cada match com a H ground-truth: ||H·p1 - p2|| em px (M,)."""
    p1, p2 = matches_to_points(f1, f2, matches)
    return np.linalg.norm(project(p1, H) - p2, axis=1)


def gt_correspondences(p1: np.ndarray, pk: np.ndarray, H: np.ndarray, shape1, shapek, eps: float
                       ) -> tuple[int, int, int]:
    """Correspondências ground-truth entre keypoints (D17).

    Considera só os pontos na zona comum: p1 cuja projeção H·p1 cai dentro da imagem k e pk cuja
    projeção inversa cai dentro da imagem 1. Dois pontos podem corresponder se ||H·p1 - pk|| < eps.
    N_corresp = emparelhamento 1-para-1 MÁXIMO desse grafo bipartido (garante N_correct <= N_corresp).
    Devolve (N_corresp, n1_visíveis, nk_visíveis).
    """
    if len(p1) == 0 or len(pk) == 0:
        return 0, 0, 0
    p1w = project(p1, H)
    vis1 = inside(p1w, shapek)
    visk = inside(project(pk, np.linalg.inv(H)), shape1)
    a, b = p1w[vis1], pk[visk]
    n1, nk = len(a), len(b)
    if n1 == 0 or nk == 0:
        return 0, n1, nk
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)   # (n1, nk), <= 2000x2000
    rows, cols = np.nonzero(d < eps)
    if len(rows) == 0:
        return 0, n1, nk
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n1, nk))
    n_corr = int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())
    return n_corr, n1, nk


# ---------------------------------------------------------------- métricas de um par

def evaluate_pair(f1: Features, fk: Features, mr: MatchResult, H_gt: np.ndarray,
                  shape1, shapek, cfg: dict) -> dict:
    """Métricas de um par (img1 -> imgk). Devolve um dict "plano" = uma linha do CSV.

    Contagens (denominadores, D02): N_features = nº de keypoints da img1 (todos);
    N_corresp e repetibilidade só na zona comum. Quando N_putative = 0 a precisão é 0
    (o método falhou); quando N_corresp = 0 o recall e a repetibilidade ficam NaN (D18).
    """
    eps = cfg["eval"]["eps_px"]
    n_corr, n1_vis, nk_vis = gt_correspondences(f1.points(), fk.points(), H_gt, shape1, shapek, eps)
    err = match_errors(f1, fk, mr.matches, H_gt)
    n_feat, n_put = f1.n, len(mr.matches)
    n_correct = int((err < eps).sum())

    t_det = (f1.t_det + fk.t_det) / 2          # média por imagem
    t_desc = (f1.t_desc + fk.t_desc) / 2
    return {
        # deteção
        "N_kps1": f1.n, "N_kpsk": fk.n,
        "N_detected1": f1.meta.get("n_detected", f1.n),
        "N_vis1": n1_vis, "N_visk": nk_vis,
        "N_corresp": n_corr,
        "repeatability": n_corr / min(n1_vis, nk_vis) if min(n1_vis, nk_vis) > 0 else np.nan,
        # matching
        "N_features": n_feat, "N_putative": n_put, "N_correct": n_correct,
        "PMR": n_put / n_feat if n_feat else 0.0,
        "precision": n_correct / n_put if n_put else 0.0,
        "matching_score": n_correct / n_feat if n_feat else 0.0,
        "recall": n_correct / n_corr if n_corr else np.nan,
        "mean_err_correct_px": float(err[err < eps].mean()) if n_correct else np.nan,
        # custo (ms por imagem; matching por par)
        "t_det_ms": t_det, "t_desc_ms": t_desc, "t_match_ms": mr.t_match,
        "t_total_ms": 2 * (t_det + t_desc) + mr.t_match,
        "joint_timing": bool(f1.meta.get("joint_timing", False)),
        # descritor
        "desc_dim": f1.desc_dim, "desc_bytes_per_kp": f1.desc_bytes_per_kp,
        "desc_mem_kb": f1.desc_bytes / 1024, "binary": f1.is_binary,
        "matcher": mr.method,
    }


# ---------------------------------------------------------------- corrida no HPatches

def _warm_up(pipelines: list[Pipeline], gray1, grayk, cfg: dict) -> None:
    """Corre cada pipeline uma vez sem medir (D06): a 1.ª chamada inclui inicializações internas
    (ex.: ORB 961 ms vs 7 ms) que não fazem parte do custo do método."""
    for p in pipelines:
        p.match(p.extract(gray1, cfg), p.extract(grayk, cfg), cfg)


def run_hpatches(pipelines: list[Pipeline], cfg: dict, seqs=None, out_csv: str | Path | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Corre todos os pipelines em todos os pares 1->k do HPatches (D04).

    As features da img1 são calculadas uma vez por sequência e pipeline e reutilizadas para k = 2..6.
    Um erro num par/pipeline não pára a corrida: fica registado na coluna "error".
    Devolve um DataFrame com uma linha por (pipeline, seq, k); grava em out_csv se dado.
    """
    root = cfg["paths"]["hpatches"]
    cats = tuple(cfg["eval"]["categories"])
    ks = tuple(cfg["eval"]["pairs_k"])
    seq_list = list_sequences(root, cats, seqs)
    rows, cache_seq, cache_f1 = [], None, {}
    t_start = time.perf_counter()
    warmed = False

    for pr in iter_hpatches_pairs(root, cats, seqs, ks):
        if not warmed:
            _warm_up(pipelines, pr.gray1, pr.grayk, cfg)
            warmed = True
        if pr.seq != cache_seq:
            cache_seq, cache_f1 = pr.seq, {}
            if verbose:
                i = seq_list.index(pr.seq) + 1
                print(f"[{i:3d}/{len(seq_list)}] {pr.seq}  ({time.perf_counter() - t_start:6.0f} s)", flush=True)
        for p in pipelines:
            base = {"pipeline": p.name, "seq": pr.seq, "category": pr.category, "k": pr.k}
            try:
                if p.name not in cache_f1:
                    cache_f1[p.name] = p.extract(pr.gray1, cfg)
                f1 = cache_f1[p.name]
                fk = p.extract(pr.grayk, cfg)
                mr = p.match(f1, fk, cfg)
                rows.append(base | evaluate_pair(f1, fk, mr, pr.H, pr.gray1.shape, pr.grayk.shape, cfg)
                            | {"error": ""})
            except Exception as e:   # regista e continua
                rows.append(base | {"error": f"{type(e).__name__}: {e}"})
                if verbose:
                    print(f"   ERRO {p.name} {pr.seq} 1->{pr.k}: {e}", flush=True)

    df = pd.DataFrame(rows)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------- agregação em tabelas

_COUNT_COLS = ["N_features", "N_putative", "N_correct", "N_corresp"]
_MATCH_COLS = ["PMR", "precision", "matching_score", "recall"]


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agrega o CSV por pares em tabelas (D18: média por par, "macro"; contagens também em média).

    T1_detection   pipeline x categoria: nº kps, nº correspondências GT, repetibilidade
    T2_matching    pipeline x categoria: contagens (denominadores) + PMR, precision, MS, recall
    T5_cost        pipeline: tempos (ms), dimensão, bytes/descritor, memória
    F1_by_k        pipeline x categoria x k: precision e MS (degradação com a dificuldade)
    errors         linhas com erro (deve estar vazia)
    """
    ok = df[df["error"] == ""]
    g = ok.groupby(["pipeline", "category"])
    t1 = g[["N_kps1", "N_vis1", "N_corresp", "repeatability"]].mean()
    t2 = g[_COUNT_COLS + _MATCH_COLS].mean()
    t2.insert(0, "n_pairs", g.size())
    t5 = ok.groupby("pipeline")[["t_det_ms", "t_desc_ms", "t_match_ms", "t_total_ms",
                                 "desc_dim", "desc_bytes_per_kp", "desc_mem_kb"]].mean()
    t5["joint_timing"] = ok.groupby("pipeline")["joint_timing"].any()
    f1 = ok.groupby(["pipeline", "category", "k"])[["precision", "matching_score"]].mean()
    errors = df[df["error"] != ""]
    return {"T1_detection": t1, "T2_matching": t2, "T5_cost": t5, "F1_by_k": f1, "errors": errors}


def save_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path, prefix: str = "") -> None:
    """Grava cada tabela em <out_dir>/<prefix><nome>.csv (com 4 casas decimais)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{prefix}{name}.csv", float_format="%.4f")
