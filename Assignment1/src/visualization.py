"""Visualizações: keypoints, matches certos/errados, gráficos F1/F2, panoramas.  [A + B]

Figuras matplotlib com eixos em píxeis e identificação de cada imagem (D21):
  plot_keypoints  — posição dos keypoints numa imagem, marcador igual para todos os métodos
  plot_matches    — duas imagens lado a lado; linhas verdes/vermelhas segundo uma máscara booleana
                    (matches corretos vs errados com a H GT, ou inliers vs outliers do RANSAC)
  plot_curves     — gráfico de linhas (estudos de parâmetros, curvas por nível)
  plot_test_pattern — pares de teste de um descritor binário (BRIEF próprio)
  plot_scatter    — dispersão com um ponto por método (compromisso exatidão vs tempo)
  save_figure     — grava (PNG ou JPEG pela extensão) e fecha a figura
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")            # sem janelas: só gravar ficheiros
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from interface import Features  # noqa: E402

GREEN = "#00b400"
RED = "#ff2020"
YELLOW = "#ffd400"


def _show(ax, img: np.ndarray, title: str) -> None:
    """Mostra uma imagem (cinzento ou BGR) com eixos em píxeis. Os centros dos píxeis ficam nas
    coordenadas inteiras, a mesma convenção dos keypoints do OpenCV."""
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.tick_params(labelsize=8)


def plot_keypoints(img: np.ndarray, f: Features, title: str, img_label: str):
    """Figura com TODOS os keypoints de f sobre img, com o mesmo marcador para todos os métodos.
    O marcador não representa escala nem orientação: comparam-se número e distribuição espacial."""
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7 * h / w + 0.5), layout="constrained")
    _show(ax, img, img_label)
    pts = f.points()
    ax.scatter(pts[:, 0], pts[:, 1], s=14, facecolors="none", edgecolors=YELLOW, linewidths=0.8)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_matches(img1: np.ndarray, img2: np.ndarray, f1: Features, f2: Features, matches,
                 good: np.ndarray | None, label1: str, label2: str, title: str,
                 good_label: str = "correto", bad_label: str = "errado",
                 max_draw: int | None = 150, seed: int = 0):
    """Figura com as duas imagens lado a lado (eixos próprios) e uma linha por match.

    good: máscara booleana (len(matches)) -> verde se True, vermelho se False; None -> tudo verde.
    max_draw: nº máximo de linhas (amostra aleatória com seed fixa, mantém a proporção certo/errado).
    Os errados são desenhados por cima. Devolve (fig, nº de linhas desenhadas).
    """
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 13 * h / w + 0.9), layout="constrained")
    _show(ax1, img1, label1)
    _show(ax2, img2, label2)
    ax2.yaxis.tick_right()                      # eixo y da 2.ª imagem à direita: as linhas não o tapam
    ax2.yaxis.set_label_position("right")

    n = len(matches)
    good = np.ones(n, bool) if good is None else np.asarray(good, bool)
    idx = np.arange(n)
    if max_draw is not None and n > max_draw:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_draw, replace=False))
    for want in (True, False):                       # verdes primeiro, vermelhos por cima
        col = GREEN if want else RED
        sel = idx[good[idx] == want]
        if len(sel) == 0:
            continue
        p = np.array([f1.kps[matches[i].queryIdx].pt for i in sel])
        q = np.array([f2.kps[matches[i].trainIdx].pt for i in sel])
        ax1.scatter(p[:, 0], p[:, 1], s=12, facecolors="none", edgecolors=col, linewidths=0.8)
        ax2.scatter(q[:, 0], q[:, 1], s=12, facecolors="none", edgecolors=col, linewidths=0.8)
        for a, b in zip(p, q):
            fig.add_artist(ConnectionPatch(xyA=a, coordsA=ax1.transData, xyB=b, coordsB=ax2.transData,
                                           color=col, linewidth=0.7, alpha=0.85))
    handles = [Line2D([], [], color=GREEN, lw=2, label=good_label)]
    if good is not None and (~good).any():
        handles.append(Line2D([], [], color=RED, lw=2, label=bad_label))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=12)
    return fig, len(idx)


def plot_test_pattern(pairs: np.ndarray, patch_size: int, title: str):
    """Padrão de testes de um descritor binário: um segmento por par (x_i, y_i), em coordenadas do
    patch (origem no centro, píxeis). pairs: (n, 4) com (x1, y1, x2, y2)."""
    half = patch_size / 2
    fig, ax = plt.subplots(figsize=(5, 5.3), layout="constrained")
    for x1, y1, x2, y2 in pairs:
        ax.plot([x1, x2], [y1, y2], color="#1f77b4", linewidth=0.7, alpha=0.7)
    ax.scatter(pairs[:, [0, 2]].ravel(), pairs[:, [1, 3]].ravel(), s=4, color="black", zorder=3)
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(half + 1, -half - 1)          # y para baixo, como na imagem
    ax.set_aspect("equal")
    ax.set_xlabel("x (px, relativo ao centro do patch)")
    ax.set_ylabel("y (px, relativo ao centro do patch)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig


def plot_curves(series: list[dict], xlabel: str, ylabel: str, title: str, hlines: list[dict] = (),
                logx: bool = False, logy: bool = False, xticks=None, xticklabels=None, ylim=None):
    """Gráfico de linhas. series: [{"x", "y", "label", "style"(opcional, ex. "-o"), "color"(opcional)}];
    hlines: [{"y", "label"}] linhas horizontais de referência (pontilhado preto);
    xticklabels: rótulos dos xticks (omissão: os valores); ylim: (mín, máx) do eixo y."""
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for s in series:
        ax.plot(s["x"], s["y"], s.get("style", "-o"), label=s["label"], color=s.get("color"),
                linewidth=1.6, markersize=5)
    for h in hlines:
        ax.axhline(h["y"], color="black", linestyle=(0, (1, 2)), linewidth=1.4, label=h["label"])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels if xticklabels is not None else [f"{v:g}" for v in xticks])
        ax.xaxis.set_minor_locator(NullLocator())     # só os ticks pedidos no eixo x
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_scatter(points: list[dict], xlabel: str, ylabel: str, title: str, logx: bool = False, ylim=None):
    """Gráfico de dispersão com um ponto por método, identificado na legenda (cor e marcador).
    points: [{"x", "y", "label", "color"(opcional), "marker"(opcional)}]."""
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for p in points:
        ax.scatter(p["x"], p["y"], s=80, color=p.get("color"), marker=p.get("marker", "o"),
                   label=p["label"], zorder=3)
    ax.legend(fontsize=8)
    if logx:
        ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150, jpeg_quality: int = 92) -> Path:
    """Grava a figura (cria a pasta) e fecha-a. Formato pela extensão; .jpg usa jpeg_quality
    (fotografias ficam ~4-5x mais leves do que em PNG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"pil_kwargs": {"quality": jpeg_quality}} if path.suffix.lower() in (".jpg", ".jpeg") else {}
    fig.savefig(path, dpi=dpi, **extra)
    plt.close(fig)
    return path
