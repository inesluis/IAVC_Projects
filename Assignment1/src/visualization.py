"""Visualizações: keypoints, matches certos/errados, gráficos F1/F2, panoramas.  [A + B]

Funções de desenho (devolvem imagens BGR uint8, reutilizáveis no stitching):
  draw_keypoints  — keypoints com escala e orientação
  draw_matches    — duas imagens lado a lado; linhas verdes/vermelhas segundo uma máscara booleana
                    (matches corretos vs errados com a H GT, ou inliers vs outliers do RANSAC)
Funções de figura (matplotlib, para o relatório):
  panel_grid, save_figure
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")            # sem janelas: só gravar ficheiros
import matplotlib.pyplot as plt  # noqa: E402

from interface import Features  # noqa: E402

GREEN = (0, 200, 0)    # BGR
RED = (0, 0, 255)
YELLOW = (0, 220, 255)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Converte cinzento para BGR (cópia); imagens BGR são copiadas."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


def strongest(kps, n: int | None) -> list:
    """Os n keypoints com maior resposta (todos se n=None)."""
    kps = list(kps)
    return kps if n is None or len(kps) <= n else sorted(kps, key=lambda k: -k.response)[:n]


def draw_keypoints(img: np.ndarray, f: Features, max_draw: int | None = 500) -> np.ndarray:
    """Desenha os keypoints (círculo = escala, raio = orientação, quando o detetor os dá).
    Para legibilidade desenham-se só os max_draw mais fortes (D21)."""
    return cv2.drawKeypoints(to_bgr(img), strongest(f.kps, max_draw), None, color=YELLOW,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches(img1: np.ndarray, img2: np.ndarray, f1: Features, f2: Features, matches,
                 good: np.ndarray | None = None, max_draw: int | None = 200, seed: int = 0) -> np.ndarray:
    """Duas imagens lado a lado com linhas entre pontos emparelhados.

    good: máscara booleana (len(matches)) -> verde se True, vermelho se False; None -> tudo verde.
    max_draw: nº máximo de linhas (amostra aleatória com seed fixa, mantém a proporção certo/errado).
    Os errados são desenhados por cima para ficarem visíveis.
    """
    a, b = to_bgr(img1), to_bgr(img2)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b
    off = a.shape[1]

    n = len(matches)
    good = np.ones(n, bool) if good is None else np.asarray(good, bool)
    idx = np.arange(n)
    if max_draw is not None and n > max_draw:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_draw, replace=False))
    for want in (True, False):                      # verdes primeiro, vermelhos por cima
        for i in idx[good[idx] == want]:
            m = matches[i]
            p = tuple(int(round(v)) for v in f1.kps[m.queryIdx].pt)
            q = f2.kps[m.trainIdx].pt
            q = (int(round(q[0])) + off, int(round(q[1])))
            col = GREEN if want else RED
            cv2.line(canvas, p, q, col, 1, cv2.LINE_AA)
            cv2.circle(canvas, p, 3, col, 1, cv2.LINE_AA)
            cv2.circle(canvas, q, 3, col, 1, cv2.LINE_AA)
    return canvas


def panel_grid(images: list[np.ndarray], titles: list[str], ncols: int = 2,
               panel_width_in: float = 8.0, suptitle: str = ""):
    """Figura matplotlib com as imagens BGR em grelha, cada uma com o seu título."""
    nrows = int(np.ceil(len(images) / ncols))
    aspect = images[0].shape[0] / images[0].shape[1]
    title_in = 0.35                              # espaço vertical por título, para não sobrepor imagens
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel_width_in * ncols, (panel_width_in * aspect + title_in) * nrows + 0.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, t in zip(axes.ravel(), images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 100) -> Path:
    """Grava a figura (cria a pasta) e fecha-a."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
