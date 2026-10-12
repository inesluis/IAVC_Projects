"""Testes de sanidade do código de avaliação, em casos em que a resposta certa é conhecida.

Correr com:  python tests/test_sanity.py     (precisa das imagens GRAF em data/graf/)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brief_own import STRATEGIES, make_pairs as make_brief_pairs  # noqa: E402
from config import get_config, set_seeds  # noqa: E402
from datasets import ImagePair  # noqa: E402
from evaluation import evaluate_pair  # noqa: E402
from methods import BRIEF_OWN_METHODS, CLASSIC_METHODS, extract_features, match_features  # noqa: E402
from studies.rot_scale_study import make_pairs  # noqa: E402

ALL_METHODS = CLASSIC_METHODS + BRIEF_OWN_METHODS


def evaluate(method, pair, cfg):
    f1 = extract_features(method, pair.gray1, cfg)
    fk = extract_features(method, pair.grayk, cfg)
    matches, t_match = match_features(method, f1, fk, cfg)
    return evaluate_pair(f1, fk, matches, t_match, pair, cfg)


def test_same_image(cfg, pairs):
    """Uma imagem comparada consigo própria (H = identidade): repetibilidade e precisão têm de dar 1.
    O recall pode ficar ligeiramente abaixo de 1: se dois pontos diferentes tiverem descritores
    iguais, o ratio test rejeita-os (acontece no BRIEF-GV, que é menos distintivo)."""
    gray = pairs[("scale", 1.0)].gray1
    pair = ImagePair("teste", "teste", 1, gray, gray, np.eye(3))
    for method in ALL_METHODS:
        r = evaluate(method, pair, cfg)
        assert r["repeatability"] == 1.0, (method, r["repeatability"])
        assert r["precision"] == 1.0, (method, r["precision"])
        assert r["recall"] > 0.99, (method, r["recall"])


def test_known_homography(cfg, pairs):
    """Imagem escalada 0,9x com homografia conhecida: caso fácil, a precisão tem de ser alta
    e as métricas têm de ser coerentes entre si."""
    pair = pairs[("scale", 0.9)]
    for method in ALL_METHODS:
        r = evaluate(method, pair, cfg)
        assert r["precision"] > 0.9, (method, r["precision"])
        assert r["N_correct"] <= r["N_corresp"], method                       # logo recall <= 1
        assert np.isclose(r["matching_score"], r["PMR"] * r["precision"]), method


def test_brief_pairs(cfg):
    """Os pares do BRIEF próprio são sempre os mesmos e ficam dentro do patch."""
    half = cfg["brief_own"]["patch_size"] // 2
    for strategy in STRATEGIES:
        pairs = make_brief_pairs(strategy, cfg)
        assert pairs.shape == (cfg["brief_own"]["n_bits"], 4)
        assert np.array_equal(pairs, make_brief_pairs(strategy, cfg))
        assert pairs.min() >= -half and pairs.max() <= half


if __name__ == "__main__":
    cfg = get_config()
    set_seeds(cfg)
    synthetic = {(p.category, p.k): p for p in make_pairs(cfg)}
    test_same_image(cfg, synthetic)
    test_known_homography(cfg, synthetic)
    test_brief_pairs(cfg)
    print("Todos os testes passaram.")
