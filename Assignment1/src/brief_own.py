# BRIEF implementado, com 4 estratégias de amostragem (G I, G II, G III, G V) 
# Segue Calonder et al., "BRIEF: Binary Robust Independent Elementary Features", ECCV 2010, secção 3.
""" 
Num patch S x S suavizado, centrado no keypoint, cada bit do descritor é um teste:

    bit = 1 se p(x) < p(y), senão 0

As estratégias diferem na escolha dos pares de píxeis (x, y) 
"""

import cv2
import numpy as np
STRATEGIES = ["GI", "GII", "GIII", "GV"]

def polar_grid(patch_size: int, n_points: int, n_radii: int) -> np.ndarray:
    # Pontos de uma grelha polar: n_radii anéis até S/2, cada um com n_points / n_radii ângulos 
    # O artigo só diz "; o nº de anéis é escolha nossa 
    n_angles = n_points // n_radii
    radii = (np.arange(n_radii) + 1) * (patch_size / 2) / n_radii
    angles = np.arange(n_angles) * 2 * np.pi / n_angles
    r, a = np.meshgrid(radii, angles, indexing="ij")
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=-1).reshape(-1, 2)

def make_pairs(strategy: str, cfg: dict) -> np.ndarray:
    # Gera os pares de teste de uma estratégia: array (n_bits, 4) com (x1, y1, x2, y2) em píxeis relativos ao centro do patch  
    # A seed é fixa: os mesmos pares têm de ser usados em todas as imagens, senão os descritores não são comparáveis 
     
    p = cfg["brief_own"]
    S, n = p["patch_size"], p["n_bits"]
    half = S / 2
    rng = np.random.default_rng(cfg["seed"])

    if strategy == "GI":        # X e Y uniformes no patch
        x = rng.uniform(-half, half, (n, 2))
        y = rng.uniform(-half, half, (n, 2))
    elif strategy == "GII":     # X e Y gaussianos, variância S^2/25 (desvio padrão S/5)
        x = rng.normal(0, S / 5, (n, 2))
        y = rng.normal(0, S / 5, (n, 2))
    elif strategy == "GIII":    # X gaussiano; Y gaussiano centrado em X, variância S^2/100
        x = np.clip(rng.normal(0, S / 5, (n, 2)), -half, half)
        y = rng.normal(x, S / 10)
    elif strategy == "GV":      # X no centro; Y em todos os pontos de uma grelha polar
        x = np.zeros((n, 2))
        y = polar_grid(S, n, p["polar_radii"])
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    
    # pontos fora do patch ficam na borda; depois arredonda-se ao píxel mais próximo
    pairs = np.hstack([np.clip(x, -half, half), np.clip(y, -half, half)])
    return np.rint(pairs).astype(np.int32)

def compute_brief(gray, kps, pairs: np.ndarray, cfg: dict):
    # Calculo do descritor BRIEF dos keypoints. Devolve (keypoints mantidos, descritores).
    # Os keypoints a menos de 28 px da borda são removidos (metade do patch + raio do kernel),
    # como faz o BRIEF do OpenCV. Os descritores são uint8 (N, n_bits / 8).
     
    p = cfg["brief_own"]
    margin = p["patch_size"] // 2 + p["blur_ksize"] // 2
    h, w = gray.shape

    pts = np.rint([k.pt for k in kps]).astype(int).reshape(-1, 2)
    inside = (pts[:, 0] >= margin) & (pts[:, 0] < w - margin) & (pts[:, 1] >= margin) & (pts[:, 1] < h - margin)
    kps = [k for k, ok in zip(kps, inside) if ok]
    pts = pts[inside]
    if len(kps) == 0:
        return [], None

    # suavizar antes de comparar (float para o arredondamento não criar empates)
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (p["blur_ksize"], p["blur_ksize"]), p["blur_sigma"])

    # para cada keypoint (linhas) e cada par (colunas), comparar os dois píxeis
    x, y = pts[:, 0:1], pts[:, 1:2]
    x1, y1, x2, y2 = pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3]
    bits = smooth[y + y1, x + x1] < smooth[y + y2, x + x2]

    return kps, np.packbits(bits, axis=1)   # 8 bits por byte, para usar a distância de Hamming