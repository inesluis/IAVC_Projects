"""BRIEF implementado por nós: estratégias de amostragem G I–V (Calonder et al.).  [Pessoa A]

Segue Calonder, Lepetit, Strecha, Fua, "BRIEF: Binary Robust Independent Elementary Features",
ECCV 2010, secção 3 (e a versão TPAMI 2012). Para um patch S x S suavizado, centrado no keypoint:

    bit i = 1 se p(x_i) < p(y_i), senão 0          (Eq. 1 do artigo)

O que muda entre estratégias é a escolha dos n_d pares (x_i, y_i) (secção 3.2). Ver DECISOES.md, D25.
Implementadas: G I, G II, G III e G V. Sem orientação nem escala, como o BRIEF original.
"""

import functools

import cv2
import numpy as np

STRATEGIES = ("GI", "GII", "GIII", "GV")


def polar_grid(patch_size: int, n_points: int, n_radii: int) -> np.ndarray:
    """Grelha polar "grosseira" com n_points pontos: n_radii anéis igualmente espaçados até S/2,
    cada um com n_points / n_radii ângulos. O artigo não especifica a grelha: é escolha nossa (D25)."""
    if n_points % n_radii:
        raise ValueError(f"n_bits ({n_points}) tem de ser múltiplo de polar_radii ({n_radii})")
    n_angles = n_points // n_radii
    radii = (np.arange(n_radii) + 1) * (patch_size / 2) / n_radii
    angles = np.arange(n_angles) * 2 * np.pi / n_angles
    r, a = np.meshgrid(radii, angles, indexing="ij")
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1).reshape(-1, 2)


@functools.lru_cache(maxsize=None)
def make_pairs(strategy: str, patch_size: int = 48, n_bits: int = 256, seed: int = 0,
               polar_radii: int = 8) -> np.ndarray:
    """Gera os n_bits pares de teste de uma estratégia. Devolve int32 (n_bits, 4) com
    (x1, y1, x2, y2) em píxeis, relativos ao centro do patch, dentro de [-S/2, S/2].

    GI    X, Y ~ Uniforme(-S/2, S/2)
    GII   X, Y ~ Gaussiana(0, S^2/25)                      (a usada pelos autores)
    GIII  X ~ Gaussiana(0, S^2/25), Y ~ Gaussiana(X, S^2/100)   (pares curtos)
    GV    X = (0, 0), Y percorre todos os pontos de uma grelha polar com n_bits pontos (não aleatória)

    Coordenadas fora do patch são encostadas à borda (o artigo refere-o para a G III; aplicamos também
    à G II) e depois arredondadas ao píxel. Determinístico para a mesma seed: os MESMOS pares têm de
    ser usados nas duas imagens.
    """
    S, half = float(patch_size), patch_size / 2
    rng = np.random.default_rng(seed)
    if strategy == "GI":
        x = rng.uniform(-half, half, (n_bits, 2))
        y = rng.uniform(-half, half, (n_bits, 2))
    elif strategy == "GII":
        x = rng.normal(0.0, S / 5, (n_bits, 2))
        y = rng.normal(0.0, S / 5, (n_bits, 2))
    elif strategy == "GIII":
        x = np.clip(rng.normal(0.0, S / 5, (n_bits, 2)), -half, half)
        y = rng.normal(x, S / 10)
    elif strategy == "GV":
        x = np.zeros((n_bits, 2))
        y = polar_grid(patch_size, n_bits, polar_radii)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy} (disponíveis: {STRATEGIES})")
    pairs = np.hstack([np.clip(x, -half, half), np.clip(y, -half, half)])
    return np.rint(pairs).astype(np.int32)


class BriefOwn:
    """Extrator BRIEF próprio com a mesma interface do OpenCV: compute(gray, kps) -> (kps, desc)."""

    def __init__(self, strategy: str, cfg: dict):
        p = cfg["brief_own"]
        self.patch_size = p["patch_size"]
        self.ksize = p["blur_ksize"]
        self.sigma = p["blur_sigma"]
        self.pairs = make_pairs(strategy, p["patch_size"], p["n_bits"], cfg["seed"], p["polar_radii"])
        # margem à borda: metade do patch + raio do kernel (igual ao BRIEF do OpenCV: 24 + 4 = 28 px)
        self.margin = self.patch_size // 2 + self.ksize // 2

    def compute(self, gray: np.ndarray, kps) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Devolve (kps mantidos, desc uint8 (N, n_bits/8)). Remove keypoints a menos de `margin` px
        da borda. Os bits são empacotados com np.packbits para usar NORM_HAMMING."""
        h, w = gray.shape
        pts = np.rint(np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)).astype(np.int64)
        m = self.margin
        keep = (pts[:, 0] >= m) & (pts[:, 0] < w - m) & (pts[:, 1] >= m) & (pts[:, 1] < h - m)
        kept = [k for k, ok in zip(kps, keep) if ok]
        if not kept:
            return [], None
        pts = pts[keep]
        # suavização antes dos testes (secção 3.1); float32 evita empates criados pelo arredondamento
        smooth = cv2.GaussianBlur(gray.astype(np.float32), (self.ksize, self.ksize), self.sigma)
        x, y = pts[:, 0:1], pts[:, 1:2]                     # (N, 1)
        x1, y1, x2, y2 = (self.pairs[:, i][None, :] for i in range(4))   # (1, n_bits)
        bits = smooth[y + y1, x + x1] < smooth[y + y2, x + x2]           # (N, n_bits)
        return kept, np.packbits(bits, axis=1)
