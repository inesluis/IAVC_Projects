"""Gráficos feitos a partir das tabelas já calculadas, sem repetir corridas (D26).  [Pessoa A]

F1  curvas por nível: uma figura por categoria (i, v, rot, scale) e por métrica, uma linha por método.
    O nível é o par 1->k (HPatches), o ângulo (rotação) ou o fator (escala).
F2  exatidão contra tempo: um ponto por método, para iluminação e para ponto de vista.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization import plot_curves, plot_scatter, save_figure

METRICS = {"precision": "Precisão", "matching_score": "Matching score",
           "recall": "Recall", "repeatability": "Repetibilidade"}
CATEGORY_NAMES = {"i": "iluminação (HPatches i_*)", "v": "ponto de vista (HPatches v_*)",
                  "rot": "rotação sintética (GRAF imagem 1)", "scale": "escala sintética (GRAF imagem 1)"}
X_LABELS = {"i": "Par de imagens", "v": "Par de imagens", "rot": "Ângulo de rotação (°)",
            "scale": "Fator de escala (eixo logarítmico)"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def method_styles(methods) -> dict:
    """Cor e marcador de cada método, iguais em todas as figuras."""
    colors = plt.get_cmap("tab10").colors
    return {m: {"color": colors[i], "marker": MARKERS[i]} for i, m in enumerate(methods)}


def comma(value: float) -> str:
    """Número com vírgula decimal."""
    return f"{value:g}".replace(".", ",")


def protocol_text(cfg: dict) -> str:
    return (f"Protocolo: limite de {cfg['max_keypoints']} keypoints por imagem, ratio test "
            f"{comma(cfg['matching']['ratio'])} + cross-check, match correto se o erro com a homografia "
            f"ground-truth for < {comma(cfg['eval']['eps_px'])} px.")


def level_ticks(category: str, levels):
    """Posições e rótulos do eixo x, e uma frase para a legenda sobre o que é cada nível."""
    if category in ("i", "v"):
        labels = [f"1→{int(k)}" for k in levels]
        if category == "v":
            note = "a mudança de ponto de vista aumenta de 1→2 para 1→6"
        else:
            note = "as imagens 2 a 6 são condições de iluminação diferentes, sem ordem de dificuldade garantida"
        return levels, labels, note
    if category == "rot":
        ticks = [k for k in levels if k % 30 == 0]
        return ticks, [f"{int(k)}" for k in ticks], "a imagem base rodada em torno do centro; 0° é o controlo"
    return levels, [comma(k) for k in levels], "a imagem base escalada em torno do centro; 1 é o controlo"


def plot_level_curves(df: pd.DataFrame, cfg: dict, folder: Path) -> list:
    """F1: cada métrica em função do nível, por categoria. Devolve [(ficheiro, legenda)]."""
    methods = list(df["method"].unique())
    styles = method_styles(methods)
    captions = []
    for category, category_name in CATEGORY_NAMES.items():
        data = df[df["category"] == category]
        if data.empty:
            continue
        levels = sorted(data["k"].unique())
        means = data.groupby(["method", "k"])[list(METRICS)].mean()
        n_per_level = len(data) // (len(methods) * len(levels))
        ticks, labels, note = level_ticks(category, levels)

        for column, metric_name in METRICS.items():
            series = [{"x": levels, "y": [means.loc[(m, k), column] for k in levels], "label": m,
                       "style": "-" + styles[m]["marker"], "color": styles[m]["color"]} for m in methods]
            # precisão e repetibilidade com eixo fixo 0-1; matching score e recall ajustados aos dados
            ylim = (-0.03, 1.03) if column in ("precision", "repeatability") else (-0.01, None)
            fig = plot_curves(series, xlabel=X_LABELS[category], ylabel=metric_name,
                              title=f"{metric_name} — {category_name}", logx=(category == "scale"),
                              xticks=ticks, xticklabels=labels, ylim=ylim)
            path = save_figure(fig, folder / f"F1_{column}_{category}.png")

            if category in ("i", "v"):
                how = f"Cada ponto é a média de {n_per_level} pares (um por sequência); {note}."
            else:
                how = f"Um par por nível: {note}."
            x_name = X_LABELS[category].split(" (")[0].lower()
            captions.append((path, f"{metric_name} de cada método em {category_name}, em função de: {x_name}. "
                                   f"{how} {protocol_text(cfg)}"))
    return captions


def plot_tradeoff(df: pd.DataFrame, cfg: dict, folder: Path) -> list:
    """F2: tempo total por par contra matching score e contra precisão, para i e v."""
    methods = list(df["method"].unique())
    styles = method_styles(methods)
    captions = []
    for category in ("i", "v"):
        data = df[df["category"] == category]
        if data.empty:
            continue
        means = data.groupby("method")[["t_total_ms", "matching_score", "precision"]].mean()
        n_pairs = len(data) // len(methods)
        for column in ("matching_score", "precision"):
            points = [{"x": means.loc[m, "t_total_ms"], "y": means.loc[m, column], "label": m, **styles[m]}
                      for m in methods]
            fig = plot_scatter(points, xlabel="Tempo total por par (ms, eixo logarítmico)", ylabel=METRICS[column],
                               title=f"{METRICS[column]} vs tempo — {CATEGORY_NAMES[category]}", logx=True)
            path = save_figure(fig, folder / f"F2_{column}_vs_time_{category}.png")
            captions.append((path, f"Compromisso entre exatidão e custo em {CATEGORY_NAMES[category]}: "
                                   f"{METRICS[column].lower()} médio contra o tempo total médio por par, um ponto "
                                   f"por método (média de {n_pairs} pares). Tempo total = 2 × (deteção + descrição) "
                                   f"+ matching, em ms, eixo logarítmico. {protocol_text(cfg)}"))
    return captions
