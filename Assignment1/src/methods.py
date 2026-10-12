"""Lista dos métodos avaliados e as duas funções que os correm: extrair e emparelhar.

Um método é um nome, por exemplo "SIFT" ou "FAST+BRIEF". As experiências chamam sempre
extract_features() e match_features(), por isso para acrescentar um método basta acrescentar
o nome à lista e um ramo em cada função.

# PENDENTE (2026-10-08): métodos aprendidos (SuperPoint+NN, SuperPoint+LightGlue, SIFT+LightGlue).
# Dependem de learned.py (Pessoa B). Ver DECISOES.md, P06.
"""

from descriptors import BRIEF_OWN_NAMES, extract
from matching import match_descriptors

# As configurações clássicas mínimas do enunciado (§4)
CLASSIC_METHODS = ["SIFT", "ORB", "KAZE", "FAST+BRIEF", "FAST+BRISK", "FAST+FREAK"]

# FAST + BRIEF próprio, uma configuração por estratégia de amostragem (D25)
BRIEF_OWN_METHODS = ["FAST+" + name for name in BRIEF_OWN_NAMES]


def extract_features(method: str, gray, cfg: dict):
    """Deteta e descreve uma imagem em cinzento com o método dado. Devolve Features."""
    return extract(method, gray, cfg)


def match_features(method: str, f1, f2, cfg: dict):
    """Emparelha as features de duas imagens. Devolve (matches, tempo em ms)."""
    return match_descriptors(f1, f2, cfg)
