# Lista dos métodos avaliados e as duas funções que os correm: extrair e emparelhar.

from descriptors import BRIEF_OWN_NAMES, extract
from learned import extract_sift_lg, extract_superpoint, match_lightglue
from matching import match_descriptors

# As configurações clássicas mínimas do enunciado
CLASSIC_METHODS = ["SIFT", "ORB", "KAZE", "FAST+BRIEF", "FAST+BRISK", "FAST+FREAK"]

# Métodos aprendidos. Nos nomes "A+B", A é o extrator e B o matcher: NN é o matching de descritores
# dos clássicos (ratio test + cross-check) e LightGlue é o matcher aprendido
LEARNED_METHODS = ["SuperPoint+NN", "SuperPoint+LightGlue", "SIFT+LightGlue"]

ALL_METHODS = CLASSIC_METHODS + LEARNED_METHODS

# FAST + BRIEF próprio, uma configuração por estratégia de amostragem
BRIEF_OWN_METHODS = ["FAST+" + name for name in BRIEF_OWN_NAMES]

def extract_features(method: str, gray, cfg: dict):
    # Deteta e descreve uma imagem em cinzento com o método dado. Devolve Features
    if method.startswith("SuperPoint"):
        return extract_superpoint(gray, cfg)
    if method == "SIFT+LightGlue":
        return extract_sift_lg(gray, cfg)
    return extract(method, gray, cfg)

def match_features(method: str, f1, f2, cfg: dict):
    # Emparelha as features de duas imagens. Devolve (matches, tempo em ms)
    if method.endswith("LightGlue"):
        return match_lightglue(f1, f2, cfg)
    return match_descriptors(f1, f2, cfg)
