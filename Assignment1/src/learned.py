# Métodos aprendidos: SuperPoint, SIFT do LightGlue e os matchers LightGlue e SuperGlue  

import time 
import cv2
import torch
from lightglue import SIFT, LightGlue, SuperPoint
from transformers import SuperGlueForKeypointMatching
from interface import Features
 
EXTRACTORS = {"superpoint": SuperPoint, "sift": SIFT} 
METHOD_NAMES = {"superpoint": "SuperPoint", "sift": "SIFT-LG"}

_models = {} 

def get_device(cfg: dict) -> str: 
    return cfg["learned"]["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
 
def get_extractor(kind: str, cfg: dict):
    # Extrator "superpoint" ou "sift" 
    key = (kind, get_device(cfg), cfg["max_keypoints"])
    if key not in _models:
        model = EXTRACTORS[kind](max_num_keypoints=cfg["max_keypoints"])
        _models[key] = model.eval().to(get_device(cfg))
    return _models[key]
 
def get_matcher(kind: str, cfg: dict):
    # LightGlue com os pesos treinados para o extrator "superpoint" ou "sift" 
    params = cfg["learned"]["lightglue"]
    key = ("lightglue", kind, get_device(cfg), tuple(sorted(params.items())))
    if key not in _models:
        _models[key] = LightGlue(features=kind, **params).eval().to(get_device(cfg))
    return _models[key]
 
def synchronize(device: str) -> None: 
    if device.startswith("cuda"):
        torch.cuda.synchronize()
 
@torch.no_grad()
def extract_learned(kind: str, gray, cfg: dict) -> Features:
    # Deteta e descreve uma imagem em cinzento (uint8) com o extrator dado 
    device = get_device(cfg)
    extractor = get_extractor(kind, cfg)
    image = (torch.from_numpy(gray).float()[None] / 255.0).to(device)   # [1, H, W], valores em [0, 1]

    synchronize(device)
    t0 = time.perf_counter()
    try:
        raw = extractor.extract(image, resize=cfg["learned"]["resize"])
    except ValueError:
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
    # SuperPoint: descritores float32 (N, 256), comparados com a distância L2 
    return extract_learned("superpoint", gray, cfg)

def extract_sift_lg(gray, cfg: dict) -> Features:
    # SIFT do pacote LightGlue: descritores RootSIFT float32 (N, 128) 
    return extract_learned("sift", gray, cfg)
 
@torch.no_grad()
def match_lightglue(f1: Features, f2: Features, cfg: dict): 
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

def get_superglue(cfg: dict):
    # SuperGlue com os pesos e os parâmetros do config
    if "superglue" not in _models:
        params = cfg["learned"]["superglue"]
        model = SuperGlueForKeypointMatching.from_pretrained(params["weights"])
        model.config.matching_threshold = params["match_threshold"]
        model.config.sinkhorn_iterations = params["sinkhorn_iterations"]
        _models["superglue"] = model.eval().to(get_device(cfg))
    return _models["superglue"]

def padded(t1, t2, n: int):
    # Junta os tensores das duas imagens num só (1, 2, n, ...), completando com zeros até n pontos
    out = t1.new_zeros((1, 2, n) + t1.shape[1:])
    out[0, 0, :len(t1)] = t1
    out[0, 1, :len(t2)] = t2
    return out

@torch.no_grad()
def match_superglue(f1: Features, f2: Features, cfg: dict):
    if not f1.kps or not f2.kps:
        return [], 0.0
    device = get_device(cfg)
    model = get_superglue(cfg)
    r1, r2 = f1.raw, f2.raw
    n1, n2 = len(f1.kps), len(f2.kps)

    synchronize(device)
    t0 = time.perf_counter()
    # O SuperGlue normaliza as coordenadas pelo tamanho da imagem (centro e 0,7 x lado maior) 
    # a função só aceita um tamanho para as duas. Os pontos da imagem 2 são levados para o referencial da
    # imagem 1 de forma a que a normalização dê o mesmo que com o tamanho da própria imagem 2.
    (w1, h1), (w2, h2) = r1["image_size"][0].tolist(), r2["image_size"][0].tolist()
    c1, c2 = r1["image_size"][0] / 2, r2["image_size"][0] / 2
    k2 = (r2["keypoints"][0] - c2) * (max(w1, h1) / max(w2, h2)) + c1

    n = max(n1, n2)
    mask = padded(torch.ones(n1, device=device), torch.ones(n2, device=device), n)
    matches, scores, _, _ = model._match_image_pair(
        padded(r1["keypoints"][0], k2, n), padded(r1["descriptors"][0], r2["descriptors"][0], n),
        padded(r1["keypoint_scores"][0], r2["keypoint_scores"][0], n), int(h1), int(w1), mask=mask)
    synchronize(device)
    t_match = (time.perf_counter() - t0) * 1000

    idx = matches[0, 0, :n1].cpu().numpy()   # para cada ponto da imagem 1: índice na imagem 2, ou -1
    conf = scores[0, 0, :n1].cpu().numpy()
    matches = [cv2.DMatch(i, int(j), float(1 - conf[i])) for i, j in enumerate(idx) if 0 <= j < n2]
    return matches, t_match