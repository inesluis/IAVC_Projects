"""Métodos aprendidos: SuperPoint, SIFT do LightGlue e o matcher LightGlue.  [Pessoa B]

Implementação e pesos pré-treinados: pacote cvg/LightGlue (Lindenberger, Sarlin e Pollefeys, ICCV 2023).
Os pesos são descarregados pelo pacote na primeira utilização.

Papéis (enunciado §2.6):
  - SuperPoint é um detetor + descritor: uma rede dá, numa só passagem, os keypoints e os descritores
    de 256 valores. Por isso não há tempos separados: t_det é o tempo total e t_desc é 0.
  - LightGlue é só um matcher: recebe os keypoints e os descritores das duas imagens e devolve os pares.
    Precisa dos tensores do extrator, que ficam guardados em Features.raw.
  - O LightGlue treinado para SIFT espera o SIFT do próprio pacote (RootSIFT, com escala e orientação),
    que não tem os mesmos parâmetros do cv2.SIFT dos métodos clássicos.

Os keypoints são devolvidos nas coordenadas da imagem original, mesmo que cfg["learned"]["resize"]
reduza a imagem antes de a passar à rede.
"""

import time

import cv2
import torch
from lightglue import SIFT, LightGlue, SuperPoint

from interface import Features

# Pesos usados por cada extrator e pelo matcher correspondente (a citar no relatório)
CHECKPOINTS = {
    "superpoint": "superpoint_v1 (Magic Leap) + superpoint_lightglue (cvg/LightGlue v0.1_arxiv)",
    "sift": "sift_lightglue (cvg/LightGlue v0.1_arxiv)",
}

EXTRACTORS = {"superpoint": SuperPoint, "sift": SIFT}

# Features.method de cada extrator; o LightGlue usa-o para escolher os pesos
METHOD_NAMES = {"superpoint": "SuperPoint", "sift": "SIFT-LG"}

_models = {}   # modelos já carregados, para não os carregar em cada imagem


def get_device(cfg: dict) -> str:
    """"cuda" ou "cpu". Com cfg["learned"]["device"] = None usa a GPU se existir."""
    return cfg["learned"]["device"] or ("cuda" if torch.cuda.is_available() else "cpu")


def get_extractor(kind: str, cfg: dict):
    """Extrator "superpoint" ou "sift", com o limite de keypoints do config."""
    key = (kind, get_device(cfg), cfg["max_keypoints"])
    if key not in _models:
        model = EXTRACTORS[kind](max_num_keypoints=cfg["max_keypoints"])
        _models[key] = model.eval().to(get_device(cfg))
    return _models[key]


def get_matcher(kind: str, cfg: dict):
    """LightGlue com os pesos treinados para o extrator "superpoint" ou "sift"."""
    params = cfg["learned"]["lightglue"]
    key = ("lightglue", kind, get_device(cfg), tuple(sorted(params.items())))
    if key not in _models:
        _models[key] = LightGlue(features=kind, **params).eval().to(get_device(cfg))
    return _models[key]


def synchronize(device: str) -> None:
    """Na GPU as operações são assíncronas: sem esperar por elas, o tempo medido fica errado."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def extract_learned(kind: str, gray, cfg: dict) -> Features:
    """Deteta e descreve uma imagem em cinzento (uint8) com o extrator dado."""
    device = get_device(cfg)
    extractor = get_extractor(kind, cfg)
    image = (torch.from_numpy(gray).float()[None] / 255.0).to(device)   # [1, H, W], valores em [0, 1]

    synchronize(device)
    t0 = time.perf_counter()
    try:
        raw = extractor.extract(image, resize=cfg["learned"]["resize"])
    except ValueError:
        # o SIFT do pacote falha quando não deteta nenhum ponto (imagem sem textura)
        if kind != "sift":
            raise
        raw = None
    synchronize(device)
    t_det = (time.perf_counter() - t0) * 1000

    method = METHOD_NAMES[kind]
    if raw is None or raw["keypoints"].shape[1] == 0:
        return Features([], None, cv2.NORM_L2, t_det, 0.0, method, 0)

    points = raw["keypoints"][0].cpu().numpy()
    desc = raw["descriptors"][0].cpu().numpy().astype("float32")
    kps = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points]
    return Features(kps, desc, cv2.NORM_L2, t_det, 0.0, method, len(kps), raw)


def extract_superpoint(gray, cfg: dict) -> Features:
    """SuperPoint: descritores float32 (N, 256), comparados com a distância L2."""
    return extract_learned("superpoint", gray, cfg)


def extract_sift_lg(gray, cfg: dict) -> Features:
    """SIFT do pacote LightGlue: descritores RootSIFT float32 (N, 128)."""
    return extract_learned("sift", gray, cfg)


@torch.no_grad()
def match_lightglue(f1: Features, f2: Features, cfg: dict):
    """Emparelha com o LightGlue as features de duas imagens. Devolve (matches, tempo em ms).

    O resultado é 1-para-1. Em cada cv2.DMatch, queryIdx é o índice na imagem 1 e trainIdx na
    imagem 2; a distância é 1 - confiança do par, para que menor continue a ser melhor.
    """
    if not f1.kps or not f2.kps:
        return [], 0.0
    device = get_device(cfg)
    kind = {name: kind for kind, name in METHOD_NAMES.items()}[f1.method]
    matcher = get_matcher(kind, cfg)

    synchronize(device)
    t0 = time.perf_counter()
    out = matcher({"image0": f1.raw, "image1": f2.raw})
    synchronize(device)
    t_match = (time.perf_counter() - t0) * 1000

    pairs = out["matches"][0].cpu().numpy()
    scores = out["scores"][0].cpu().numpy()
    matches = [cv2.DMatch(int(i), int(j), float(1 - s)) for (i, j), s in zip(pairs, scores)]
    return matches, t_match
