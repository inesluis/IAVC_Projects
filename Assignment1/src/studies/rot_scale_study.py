"""Teste sintético de robustez à rotação e à escala (categorias 'rot' e 'scale').  [Pessoa A]

A imagem base (GRAF img1) é transformada com uma rotação OU uma escala conhecidas; a H ground-truth é a
própria transformação, por isso cada fator é avaliado isolado (o HPatches v_* mistura perspetiva,
rotação e escala). Ver DECISOES.md, D20.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from datasets import load_image
from evaluation import evaluate_pairs
from interface import Pipeline


@dataclass
class SyntheticPair:
    """Par sintético compatível com ImagePair. k = nível (graus ou fator de escala)."""
    seq: str
    category: str
    k: float
    gray1: np.ndarray
    grayk: np.ndarray
    H: np.ndarray
    control: bool = False


def circular_feather(gray: np.ndarray, feather_px: float, fill: int) -> np.ndarray:
    """Mantém só o círculo inscrito na imagem, com transição suave para o cinzento `fill`.
    Assim a rotação não cria nem perde conteúdo e a borda do círculo quase não gera keypoints."""
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    radius = min(h, w) / 2 - 1
    wgt = np.clip((radius - r) / feather_px, 0.0, 1.0)
    return np.round(gray * wgt + fill * (1 - wgt)).astype(np.uint8)


def rotate(gray: np.ndarray, angle_deg: float, fill: int) -> tuple[np.ndarray, np.ndarray]:
    """Roda a imagem em torno do centro (sentido anti-horário, mesma tela, fundo `fill`).
    Devolve (imagem, H 3x3)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle_deg, 1.0)
    out = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return out, np.vstack([M, [0, 0, 1]])


def scale(gray: np.ndarray, s: float, fill: int) -> tuple[np.ndarray, np.ndarray]:
    """Escala a imagem em torno do centro, na mesma tela (s > 1: zoom com recorte; s < 1: redução
    com margem cinzenta `fill`). INTER_AREA para reduzir (evita aliasing), INTER_LINEAR para
    ampliar. Devolve (imagem, H 3x3) com a convenção de centros de pixel do cv2.resize."""
    h, w = gray.shape
    ws, hs = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    res = cv2.resize(gray, (ws, hs), interpolation=interp)
    ox, oy = (w - ws) // 2, (h - hs) // 2            # posição do canto do redimensionado (pode ser < 0)
    out = np.full((h, w), fill, dtype=np.uint8)
    # interseção entre a tela [0,w)x[0,h) e o redimensionado colocado em (ox, oy)
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + ws, w), min(oy + hs, h)
    out[y0:y1, x0:x1] = res[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    sx, sy = ws / w, hs / h
    # cv2.resize: x_dst = sx*(x_src + 0.5) - 0.5 ; depois translação (ox, oy)
    H = np.array([[sx, 0, 0.5 * sx - 0.5 + ox],
                  [0, sy, 0.5 * sy - 0.5 + oy],
                  [0, 0, 1]], dtype=np.float64)
    return out, H


def make_pairs(cfg: dict) -> list[SyntheticPair]:
    """Pares sintéticos a partir da imagem base: primeiro todos os de rotação, depois os de escala
    (cada grupo partilha a img1, por isso as features da img1 são calculadas uma vez por grupo)."""
    sc = cfg["synthetic"]
    _, gray = load_image(sc["base_image"])
    fill = int(round(float(gray.mean())))      # mesmo cinzento de fundo em todas as operações
    pairs = []
    g_rot = circular_feather(gray, sc["mask_feather_px"], fill)
    for a in sc["rotations_deg"]:
        img, H = rotate(g_rot, a, fill)
        pairs.append(SyntheticPair("graf_rot", "rot", float(a), g_rot, img, H, control=(a == 0)))
    for s in sc["scales"]:
        img, H = scale(gray, s, fill)
        pairs.append(SyntheticPair("graf_scale", "scale", float(s), gray, img, H, control=(s == 1.0)))
    return pairs


def run_rot_scale(pipelines: list[Pipeline], cfg: dict, verbose: bool = True):
    """Avalia os pipelines em todos os pares sintéticos. Devolve o DataFrame (uma linha por
    pipeline x nível), com as mesmas colunas que a avaliação HPatches."""
    return evaluate_pairs(pipelines, make_pairs(cfg), cfg, verbose, n_seqs=2)
