# Estudo do detetor FAST: efeito do threshold e da NMS e comparação com o SIFT 

import copy
import time
import numpy as np
import pandas as pd
from datasets import hpatches_pairs
from detectors import detect
from evaluation import gt_correspondences
from interface import keypoints_to_array
from visualization import plot_curves, save_figure

def detector_configs(cfg: dict) -> list[dict]:
    # Uma configuração por combinação threshold x NMS do FAST, mais o SIFT como referência 
    # Todas sem limite de pontos, para ver o comportamento real do detetor 
    configs = []
    for nms in cfg["fast_study"]["nms"]:
        for threshold in cfg["fast_study"]["thresholds"]:
            c = copy.deepcopy(cfg)
            c["max_keypoints"] = None
            c["detectors"]["FAST"]["threshold"] = threshold
            c["detectors"]["FAST"]["nonmaxSuppression"] = nms
            name = f"FAST t={threshold} NMS={'on' if nms else 'off'}"
            configs.append({"name": name, "detector": "FAST", "threshold": threshold, "nms": nms, "cfg": c})

    c = copy.deepcopy(cfg)
    c["max_keypoints"] = None
    reference = cfg["fast_study"]["reference"]
    configs.append({"name": reference, "detector": reference, "threshold": np.nan, "nms": None, "cfg": c})
    return configs

def repeatability(p1, pk, H, shape1, shapek, eps: float) -> float:
    # Fração dos pontos da zona comum que reaparecem a menos de eps px na outra imagem 
    n_corresp, n_vis1, n_visk = gt_correspondences(p1, pk, H, shape1, shapek, eps)
    if min(n_vis1, n_visk) == 0:
        return np.nan
    return n_corresp / min(n_vis1, n_visk)

def chance_repeatability(p1, pk, H, shape1, shapek, eps: float, shifts) -> float:
    values = []
    for dx, dy in shifts:
        shift = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        values.append(repeatability(p1, pk, shift @ H, shape1, shapek, eps))
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan

def run_fast_study(cfg: dict, only=None) -> pd.DataFrame:
    # Corre todas as configurações em todos os pares do HPatches. Uma linha por (configuração, par) 
    eps = cfg["eval"]["eps_px"]
    study = cfg["fast_study"]
    configs = detector_configs(cfg)
    rows = []
    current_seq = None
    detections_1 = {}     # deteções da imagem 1 da sequência atual, por configuração
    t_start = time.perf_counter()

    for pair in hpatches_pairs(cfg, only):
        if current_seq is None:                      # aquecimento: a 1.ª chamada não conta  
            for c in configs:
                detect(c["detector"], pair.gray1, c["cfg"])
        if pair.seq != current_seq:
            current_seq = pair.seq
            detections_1 = {}
            print(f"{pair.seq}  ({time.perf_counter() - t_start:.0f} s)", flush=True)

        for c in configs:
            if c["name"] not in detections_1:
                detections_1[c["name"]] = detect(c["detector"], pair.gray1, c["cfg"])
            kps1, t1 = detections_1[c["name"]]
            kpsk, tk = detect(c["detector"], pair.grayk, c["cfg"])

            row = {"config": c["name"], "detector": c["detector"], "threshold": c["threshold"], "nms": c["nms"],
                   "seq": pair.seq, "category": pair.category, "k": pair.k,
                   "N_kps_mean": (len(kps1) + len(kpsk)) / 2, "t_det_ms": (t1 + tk) / 2}
 
            row["rep_skipped"] = max(len(kps1), len(kpsk)) > study["max_points_repeatability"]
            if row["rep_skipped"]:
                row["repeatability"] = row["repeatability_chance"] = np.nan
            else:
                p1, pk = keypoints_to_array(kps1), keypoints_to_array(kpsk)
                shapes = (pair.gray1.shape, pair.grayk.shape)
                row["repeatability"] = repeatability(p1, pk, pair.H, *shapes, eps)
                row["repeatability_chance"] = chance_repeatability(p1, pk, pair.H, *shapes, eps,
                                                                   study["chance_shifts_px"])
            rows.append(row)
    return pd.DataFrame(rows)

def common_pairs(df: pd.DataFrame) -> pd.DataFrame: 
    skipped = df[df["rep_skipped"]][["seq", "k"]].drop_duplicates()
    marked = df.merge(skipped.assign(skip=True), on=["seq", "k"], how="left")
    return df[marked["skip"].isna().to_numpy()]

def summarize_fast(df: pd.DataFrame) -> pd.DataFrame: 
    all_pairs = df.groupby(["config", "category"], sort=False)
    common = common_pairs(df).groupby(["config", "category"], sort=False)
    table = pd.DataFrame({
        "n_pairs": all_pairs.size(),
        "N_kps_mean": all_pairs["N_kps_mean"].mean(),
        "t_det_ms": all_pairs["t_det_ms"].mean(),
        "n_rep_skipped": all_pairs["rep_skipped"].sum(),
        "n_pairs_rep": common.size(),
        "repeatability": common["repeatability"].mean(),
        "repeatability_chance": common["repeatability_chance"].mean(),
    })
    return table.unstack("category")

CATEGORY_NAMES = {"i": "iluminação (HPatches i_*)", "v": "ponto de vista (HPatches v_*)"}
METRICS = {   # coluna: (nome do eixo y, eixo logarítmico?)
    "repeatability": ("Repetibilidade", False),
    "N_kps_mean": ("Nº de keypoints por imagem", True),
    "t_det_ms": ("Tempo de deteção por imagem (ms)", True),
}

def plot_fast(df: pd.DataFrame, cfg: dict, folder) -> list: 
    reference = cfg["fast_study"]["reference"]
    thresholds = cfg["fast_study"]["thresholds"]
    common = common_pairs(df)
    captions = []

    for category, category_name in CATEGORY_NAMES.items():
        n_all = df[(df["category"] == category) & (df["config"] == reference)].shape[0]
        n_common = common[(common["category"] == category) & (common["config"] == reference)].shape[0]

        for column, (ylabel, logy) in METRICS.items():
            # a repetibilidade usa só os pares comuns; as outras métricas usam todos
            data = common if column == "repeatability" else df
            data = data[data["category"] == category]
            fast = data[data["detector"] == "FAST"]

            series = []
            for nms, style, label in ((True, "-o", "NMS ligada"), (False, "--s", "NMS desligada")):
                means = fast[fast["nms"] == nms].groupby("threshold")
                series.append({"x": thresholds, "y": list(means[column].mean()[thresholds]),
                               "label": f"FAST, {label}", "style": style})
                if column == "repeatability":
                    series.append({"x": thresholds, "y": list(means["repeatability_chance"].mean()[thresholds]),
                                   "label": f"ao acaso, {label}", "style": ":" if nms else "-.", "color": "0.5"})
            reference_value = data[data["detector"] == reference][column].mean()

            title = f"FAST — {ylabel.split(' (')[0].lower()}: {category_name}"
            fig = plot_curves(series, xlabel="Threshold de intensidade do FAST", ylabel=ylabel, title=title,
                              hlines=[{"y": reference_value, "label": f"{reference} (referência)"}],
                              logx=True, logy=logy, xticks=thresholds)
            path = save_figure(fig, folder / f"fast_{column}_{category}.png")
            captions.append((path, caption(column, category_name, reference, n_all, n_common, cfg)))
    return captions

def caption(column: str, category_name: str, reference: str, n_all: int, n_common: int, cfg: dict) -> str: 
    ylabel = METRICS[column][0].split(" (")[0].lower()
    text = (f"Estudo do detetor FAST em {category_name}: {ylabel} em função do threshold de intensidade "
            f"(eixo logarítmico), com NMS ligada e desligada; linha horizontal: {reference} (valores por "
            f"omissão). Sem limite de pontos. ")
    if column == "repeatability":
        eps = f"{cfg['eval']['eps_px']:g}"
        limit = f"{cfg['fast_study']['max_points_repeatability']:,}".replace(",", " ")
        shifts = " e ".join(f"({dx:+d}, {dy:+d}) px" for dx, dy in cfg["fast_study"]["chance_shifts_px"])
        text += (f"Média sobre os {n_common} pares (de {n_all}) em que todas as configurações foram avaliadas: "
                 f"a repetibilidade não é calculada quando uma imagem tem mais de {limit} pontos, e só se "
                 f"comparam configurações nos mesmos pares. Repetibilidade = correspondências ground-truth "
                 f"1-para-1 a menos de {eps} px / mínimo dos pontos na zona comum. Cinzento: nível ao acaso, "
                 f"a mesma medida com a homografia ground-truth deslocada {shifts} (média).")
    else:
        text += f"Média sobre {n_all} pares."
    return text