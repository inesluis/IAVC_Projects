# Estudo do BRIEF próprio: figuras dos padrões de teste de cada estratégia 
 
from pathlib import Path
import numpy as np
from brief_own import STRATEGIES, make_pairs
from visualization import plot_test_pattern, save_figure

LABELS = {"GI": "G I", "GII": "G II", "GIII": "G III", "GV": "G V"}
DESCRIPTIONS = {
    "GI": "X e Y uniformes no patch, Uniforme(−S/2, S/2)",
    "GII": "X e Y gaussianos centrados no centro do patch, variância S²/25",
    "GIII": "X gaussiano (variância S²/25) e Y gaussiano centrado em X (variância S²/100)",
    "GV": "X fixo no centro e Y em todos os pontos de uma grelha polar",
}

def plot_patterns(cfg: dict, folder: Path) -> list: 
    S, n_bits = cfg["brief_own"]["patch_size"], cfg["brief_own"]["n_bits"]
    captions = []
    for strategy in STRATEGIES:
        pairs = make_pairs(strategy, cfg)
        title = f"BRIEF próprio — {LABELS[strategy]}: {n_bits} testes num patch de {S}×{S} px"
        path = save_figure(plot_test_pattern(pairs, S, title), folder / f"brief_pattern_{strategy}.png")

        # propriedades dos pares, para a legenda
        lengths = np.hypot(pairs[:, 2] - pairs[:, 0], pairs[:, 3] - pairs[:, 1])
        mean_length = f"{lengths.mean():.1f}".replace(".", ",")
        n_distinct = len(np.unique(pairs, axis=0))
        n_degenerate = int((lengths == 0).sum())

        description = DESCRIPTIONS[strategy]
        if strategy == "GV":
            n_radii = cfg["brief_own"]["polar_radii"]
            description += f" ({n_radii} anéis × {n_bits // n_radii} ângulos)"
        captions.append((path, f"Padrão de testes da estratégia {LABELS[strategy]} do BRIEF próprio: {description}. "
                               f"Cada segmento liga os dois píxeis comparados num teste (bit = 1 se p(x) < p(y)); "
                               f"{n_bits} testes, patch de {S}×{S} px, seed {cfg['seed']}. Comprimento médio dos "
                               f"pares: {mean_length} px; testes distintos: {n_distinct}; testes degenerados "
                               f"(x = y): {n_degenerate}. Coordenadas relativas ao centro do patch, em píxeis."))
    return captions
