"""Estudo do FAST: threshold × NMS -> nº de pontos, repetibilidade, tempo (tabela T6).  [Pessoa A]

Enunciado §2.3: efeito do threshold de intensidade e da non-maximum suppression (NMS), e comparação
FAST vs SIFT em repetibilidade, nº de pontos e tempo. Avalia-se só o DETETOR (sem descritor e sem
limite de pontos), no HPatches completo, separado por iluminação (i) e ponto de vista (v). Ver D24.
"""

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import iter_hpatches_pairs, list_sequences
from detectors import detect
from evaluation import gt_correspondences
from visualization import plot_curves, save_figure


def detector_configs(cfg: dict) -> list[tuple[str, dict]]:
    """[(rótulo, cfg do detetor)] para todas as combinações threshold x NMS, mais a referência.
    Todos sem limite de pontos (max_keypoints=None)."""
    out = []
    sc = cfg["fast_study"]
    for nms in sc["nms"]:
        for thr in sc["thresholds"]:
            c = copy.deepcopy(cfg)
            c["max_keypoints"] = None
            c["detectors"]["FAST"].update(threshold=thr, nonmaxSuppression=nms)
            out.append((f"FAST t={thr} NMS={'on' if nms else 'off'}",
                        {"detector": "FAST", "threshold": thr, "nms": nms, "cfg": c}))
    ref = copy.deepcopy(cfg)
    ref["max_keypoints"] = None
    out.append((sc["reference"], {"detector": sc["reference"], "threshold": np.nan, "nms": None, "cfg": ref}))
    return out


def chance_repeatability(p1: np.ndarray, pk: np.ndarray, H: np.ndarray, shape1, shapek, eps: float,
                         shifts) -> float:
    """Repetibilidade "ao acaso" (D24): a mesma medida, mas com a H ground-truth deslocada por cada
    translação em `shifts` (px), média dos resultados.

    O deslocamento mantém o nº de pontos e a sua distribuição real (aglomerados, zonas com textura),
    mas quebra a correspondência geométrica: o que ainda "repete" é coincidência. Uma fórmula para
    pontos uniformes (1 - exp(-pi eps^2 rho)) sobrestima o acaso quando os pontos estão aglomerados
    (FAST sem NMS), por isso não é usada.
    """
    vals = []
    for dx, dy in shifts:
        T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        n_corr, n1v, nkv = gt_correspondences(p1, pk, T @ H, shape1, shapek, eps)
        if min(n1v, nkv) > 0:
            vals.append(n_corr / min(n1v, nkv))
    return float(np.mean(vals)) if vals else np.nan


def run_fast_study(cfg: dict, seqs=None, verbose: bool = True) -> pd.DataFrame:
    """Corre todas as configurações em todos os pares 1->k do HPatches.
    Uma linha por (configuração, seq, k). Os pontos da img1 são detetados uma vez por seq."""
    root = cfg["paths"]["hpatches"]
    cats = tuple(cfg["eval"]["categories"])
    eps = cfg["eval"]["eps_px"]
    confs = detector_configs(cfg)
    n_seqs = len(list_sequences(root, cats, seqs))
    rows, cache_seq, cache1 = [], None, {}
    t_start = time.perf_counter()
    warmed = False

    for pr in iter_hpatches_pairs(root, cats, seqs, tuple(cfg["eval"]["pairs_k"])):
        if not warmed:                                   # D06: 1.ª chamada não conta
            for _, c in confs:
                detect(c["detector"], pr.gray1, c["cfg"])
            warmed = True
        if pr.seq != cache_seq:
            cache_seq, cache1 = pr.seq, {}
            if verbose:
                print(f"[{len(set(r['seq'] for r in rows)) + 1:3d}/{n_seqs}] {pr.seq}  "
                      f"({time.perf_counter() - t_start:6.0f} s)", flush=True)
        for label, c in confs:
            if label not in cache1:
                cache1[label] = detect(c["detector"], pr.gray1, c["cfg"])
            k1, t1 = cache1[label]
            kk, tk = detect(c["detector"], pr.grayk, c["cfg"])
            p1 = np.array([k.pt for k in k1], np.float64).reshape(-1, 2)
            pk = np.array([k.pt for k in kk], np.float64).reshape(-1, 2)
            row = {"config": label, "detector": c["detector"], "threshold": c["threshold"], "nms": c["nms"],
                   "seq": pr.seq, "category": pr.category, "k": pr.k,
                   "N_kps1": len(k1), "N_kpsk": len(kk), "N_kps_mean": (len(k1) + len(kk)) / 2,
                   "t_det_ms": (t1 + tk) / 2}
            # D24: com pontos a mais, o emparelhamento máximo pode bloquear (grafo num só aglomerado)
            skip = max(len(k1), len(kk)) > cfg["fast_study"]["max_points_repeatability"]
            if skip:
                row |= {"N_vis1": np.nan, "N_visk": np.nan, "N_corresp": np.nan,
                        "repeatability": np.nan, "repeatability_chance": np.nan}
            else:
                n_corr, n1v, nkv = gt_correspondences(p1, pk, pr.H, pr.gray1.shape, pr.grayk.shape, eps)
                row |= {"N_vis1": n1v, "N_visk": nkv, "N_corresp": n_corr,
                        "repeatability": n_corr / min(n1v, nkv) if min(n1v, nkv) > 0 else np.nan,
                        "repeatability_chance": chance_repeatability(
                            p1, pk, pr.H, pr.gray1.shape, pr.grayk.shape, eps, cfg["fast_study"]["chance_shifts_px"])}
            row["rep_skipped"] = skip
            rows.append(row)
    return pd.DataFrame(rows)


def common_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Linhas dos pares (seq, k) em que TODAS as configurações têm repetibilidade calculada (D24).
    A repetibilidade só é comparável entre configurações se a média for feita nos mesmos pares."""
    bad = df[df["rep_skipped"].astype(bool)][["seq", "k"]].drop_duplicates()
    key = df[["seq", "k"]].merge(bad.assign(_bad=True), on=["seq", "k"], how="left")["_bad"].isna().to_numpy()
    return df[key]


def summarize_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela T6: configuração x categoria. Ordem: FAST NMS on, FAST NMS off, referência.

    N_kps_mean, t_det_ms       média sobre TODOS os pares (n_pairs)
    repeatability(_chance)     média só sobre os pares comuns a todas as configurações (n_pairs_rep), D24
    n_rep_skipped              pares desta configuração sem repetibilidade por excesso de pontos
    """
    g = df.groupby(["config", "category"], sort=False)
    gc = common_pairs(df).groupby(["config", "category"], sort=False)
    t = pd.DataFrame({
        "n_pairs": g.size(),
        "N_kps_mean": g["N_kps_mean"].mean(),
        "t_det_ms": g["t_det_ms"].mean(),
        "n_rep_skipped": g["rep_skipped"].sum(),
        "n_pairs_rep": gc.size(),
        "repeatability": gc["repeatability"].mean(),
        "repeatability_chance": gc["repeatability_chance"].mean(),
    })
    return t.unstack("category")


def plot_fast(df: pd.DataFrame, cfg: dict, out_dir: Path) -> list[tuple[Path, str]]:
    """Gráficos por métrica e por categoria (1 figura por objetivo; D21). Devolve [(caminho, legenda)]."""
    eps = f"{cfg['eval']['eps_px']:g}".replace(".", ",")
    ref = cfg["fast_study"]["reference"]
    names = {"i": "iluminação (HPatches i_*)", "v": "ponto de vista (HPatches v_*)"}
    fast = df[df["detector"] == "FAST"]
    com = common_pairs(df)                       # repetibilidade: só pares comuns a todas as configurações
    cols_all, cols_rep = ["N_kps_mean", "t_det_ms"], ["repeatability", "repeatability_chance"]
    means = fast.groupby(["category", "nms", "threshold"])[cols_all].mean().join(
        com[com["detector"] == "FAST"].groupby(["category", "nms", "threshold"])[cols_rep].mean())
    refm = df[df["detector"] == ref].groupby("category")[cols_all].mean().join(
        com[com["detector"] == ref].groupby("category")[cols_rep].mean())
    n_conf = df["config"].nunique()
    n_pairs = df.groupby("category").size() // n_conf
    n_pairs_rep = com.groupby("category").size() // n_conf
    thr = sorted(fast["threshold"].unique())
    shift_txt = " e ".join(f"({dx:+g}, {dy:+g}) px" for dx, dy in cfg["fast_study"]["chance_shifts_px"]) \
        + " (média)"
    metrics = {
        "repeatability": ("Repetibilidade", False,
                          f"Repetibilidade = correspondências ground-truth 1-para-1 a < {eps} px / mínimo dos pontos "
                          f"na zona comum. Cinzento: nível ao acaso, a mesma medida com a homografia "
                          f"ground-truth deslocada {shift_txt} (mesmos pontos, correspondência geométrica "
                          f"quebrada)."),
        "N_kps_mean": ("Nº de keypoints por imagem", True, "Nº médio de keypoints por imagem (escala logarítmica)."),
        "t_det_ms": ("Tempo de deteção por imagem (ms)", True,
                     "Tempo médio de deteção por imagem, em ms (escala logarítmica)."),
    }
    out = []
    for cat in ("i", "v"):
        for col, (ylabel, logy, what) in metrics.items():
            series = []
            for nms, style in ((True, "-o"), (False, "--s")):
                y = [means.loc[(cat, nms, t), col] for t in thr]
                series.append({"x": thr, "y": y, "label": f"FAST, NMS {'ligada' if nms else 'desligada'}",
                               "style": style})
                if col == "repeatability":
                    yc = [means.loc[(cat, nms, t), "repeatability_chance"] for t in thr]
                    series.append({"x": thr, "y": yc, "label": f"ao acaso, NMS {'ligada' if nms else 'desligada'}",
                                   "style": ":" if nms else "-.", "color": "0.5"})
            hl = [{"y": refm.loc[cat, col], "label": f"{ref} (referência)"}]
            fig = plot_curves(series, xlabel="Threshold de intensidade do FAST", ylabel=ylabel,
                              title=f"FAST — {ylabel.split(' (')[0].lower()}: {names[cat]}",
                              hlines=hl, logx=True, logy=logy, xticks=thr)
            path = save_figure(fig, out_dir / f"fast_{col}_{cat}.png")
            if col == "repeatability":
                lim = f"{cfg['fast_study']['max_points_repeatability']:,}".replace(",", " ")
                over = (f"Média sobre os {n_pairs_rep[cat]} pares 1→k (de {n_pairs[cat]}) em que todas as "
                        f"configurações foram avaliadas: a repetibilidade não é calculada quando uma imagem tem "
                        f"mais de {lim} pontos, e só se comparam configurações nos mesmos pares.")
            else:
                over = f"Média sobre {n_pairs[cat]} pares 1→k."
            cap = (f"Estudo do detetor FAST em {names[cat]}: {ylabel.split(' (')[0].lower()} em função do threshold "
                   f"de intensidade (eixo logarítmico), com NMS ligada e desligada; linha horizontal: {ref} "
                   f"(valores por omissão). Sem limite de pontos. {over} {what}")
            out.append((path, cap))
    return out


def write_captions(entries, path: Path) -> Path:
    lines = ["# Legendas — estudo do FAST", "",
             "Geradas automaticamente por `python src/main.py --fast`. Os números são os desta corrida.", ""]
    for p, cap in entries:
        lines += [f"**{p.name}**", "", cap, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
