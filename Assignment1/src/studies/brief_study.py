"""Estudo do BRIEF próprio: estratégias G I–V vs BRIEF do OpenCV (tabela T7).  [Pessoa A]

Compara FAST + BRIEF do OpenCV com FAST + BRIEF próprio (G I, G II, G III, G V): mesmo detetor,
mesmo matching, mesmo protocolo; só muda o descritor. Avaliado nas 4 categorias (i, v no HPatches;
rot, scale no teste sintético). Ver DECISOES.md, D25.
"""

from pathlib import Path

import numpy as np

from brief_own import STRATEGIES, make_pairs
from descriptors import BRIEF_OWN_COMBOS
from visualization import plot_test_pattern, save_figure

# Referência (OpenCV) + uma configuração por estratégia.
BRIEF_STUDY_PIPELINES = ("FAST+BRIEF",) + BRIEF_OWN_COMBOS

_DESCRIPTION = {
    "GI": "X e Y uniformes no patch, Uniforme(−S/2, S/2)",
    "GII": "X e Y gaussianos centrados no centro do patch, variância S²/25",
    "GIII": "X gaussiano (variância S²/25) e Y gaussiano centrado em X (variância S²/100)",
    "GV": "X fixo no centro e Y em todos os pontos de uma grelha polar ({r} anéis × {a} ângulos)",
}
_LABEL = {"GI": "G I", "GII": "G II", "GIII": "G III", "GV": "G V"}


def plot_patterns(cfg: dict, out_dir: Path) -> list[tuple[Path, str]]:
    """Uma figura por estratégia com os pares de teste (como a Fig. 2 do artigo). Devolve [(caminho, legenda)]."""
    p = cfg["brief_own"]
    S, n = p["patch_size"], p["n_bits"]
    out = []
    for s in STRATEGIES:
        pairs = make_pairs(s, S, n, cfg["seed"], p["polar_radii"])
        length = np.hypot(pairs[:, 2] - pairs[:, 0], pairs[:, 3] - pairs[:, 1])
        n_unique = len(np.unique(pairs, axis=0))
        n_degenerate = int((length == 0).sum())
        fig = plot_test_pattern(pairs, S, f"BRIEF próprio — {_LABEL[s]}: {n} testes num patch de {S}×{S} px")
        path = save_figure(fig, out_dir / f"brief_pattern_{s}.png")
        desc = _DESCRIPTION[s].format(r=p["polar_radii"], a=n // p["polar_radii"])
        mean_len = f"{length.mean():.1f}".replace(".", ",")
        cap = (f"Padrão de testes da estratégia {_LABEL[s]} do BRIEF próprio: {desc}. Cada segmento liga os dois "
               f"píxeis comparados num teste (bit = 1 se p(x) < p(y)); {n} testes, patch de {S}×{S} px, "
               f"seed {cfg['seed']}. Comprimento médio dos pares: {mean_len} px; testes distintos: {n_unique}; "
               f"testes degenerados (x = y): {n_degenerate}. Coordenadas relativas ao centro do patch, em píxeis.")
        out.append((path, cap))
    return out


def write_captions(entries, path: Path) -> Path:
    lines = ["# Legendas — BRIEF próprio", "",
             "Geradas automaticamente por `python src/main.py --brief`.", ""]
    for p, cap in entries:
        lines += [f"**{p.name}**", "", cap, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
