"""Visualizações: keypoints, matches certos/errados, gráficos F1/F2, panoramas.  [A + B]

Figuras matplotlib com eixos em píxeis e identificação de cada imagem (D21):
  plot_keypoints  — posição dos keypoints numa imagem, marcador igual para todos os métodos
  plot_matches    — duas imagens lado a lado; linhas verdes/vermelhas segundo uma máscara booleana
                    (matches corretos vs errados com a H GT, ou inliers vs outliers do RANSAC)
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


def save_figure(fig, path: str | Path, dpi: int = 150, jpeg_quality: int = 92) -> Path:
    """Grava a figura (cria a pasta) e fecha-a. Formato pela extensão; .jpg usa jpeg_quality
    (fotografias ficam ~4-5x mais leves do que em PNG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"pil_kwargs": {"quality": jpeg_quality}} if path.suffix.lower() in (".jpg", ".jpeg") else {}
    fig.savefig(path, dpi=dpi, **extra)
    plt.close(fig)
    return path
