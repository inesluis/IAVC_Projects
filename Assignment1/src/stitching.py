# stitching.py — Construção de panoramas (warp + canvas + média)

import cv2
import numpy as np
from datasets import load_image_set
from homography import chain_to_reference, estimate_homography, image_corners, project_points
from methods import match_features
 
MAX_CANVAS_SIDE = 20000  

# CANVAS
def compute_canvas(images, homographies):
    corners = [project_points(image_corners(img), H) for img, H in zip(images, homographies)]
    corners = np.vstack(corners)

    x_min, y_min = np.floor(corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(corners.max(axis=0)).astype(int)
    width, height = int(x_max - x_min), int(y_max - y_min)

    if width > MAX_CANVAS_SIDE or height > MAX_CANVAS_SIDE:
        raise ValueError(f"Canvas demasiado grande ({width}x{height} px): "
                         "a homografia está provavelmente errada.")

    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float64)
    return T, (width, height)
 
# WARP + COMPOSIÇÃO 
def warp_to_canvas(image, H, T, canvas_size):
    M = T @ H
    warped = cv2.warpPerspective(image, M, canvas_size).astype(np.float32)

    ones = np.ones(image.shape[:2], dtype=np.float32)
    mask = cv2.warpPerspective(ones, M, canvas_size)
    return warped, mask

def blend_average(warped_images, masks):
    total = np.sum(warped_images, axis=0)      # (H, W, 3) soma das cores
    count = np.sum(masks, axis=0)[..., None]   # (H, W, 1) nº de imagens em cada pixel

    # Onde não há nenhuma imagem (count = 0) o panorama fica preto 
    panorama = np.divide(total, count, out=np.zeros_like(total), where=count > 1e-6)
    return np.clip(panorama, 0, 255).astype(np.uint8)

def stitch_images(images, homographies):
    # Constrói o panorama de N imagens a cores 
    # images: lista de imagens BGR
    # homographies: homographies[i] leva a imagem i para a imagem de REFERÊNCIA (para a própria referência é a identidade, np.eye(3))
     
    T, canvas_size = compute_canvas(images, homographies)
    warped_images, masks = [], []
    for image, H in zip(images, homographies):
        warped, mask = warp_to_canvas(image, H, T, canvas_size)
        warped_images.append(warped)
        masks.append(mask)

    return blend_average(warped_images, masks)

def stitch_or_none(images, pairs, ref: int):
    # Panorama com a imagem de referência dada ou None  
    if any(pair["H"] is None for pair in pairs):
        return None
    try:
        return stitch_images(images, chain_to_reference([pair["H"] for pair in pairs], ref))
    except ValueError as error:
        print(error)
        return None
 
# DAS IMAGENS ÀS HOMOGRAFIAS 
def load_panorama_images(folder, cfg: dict):
    max_side = cfg["stitching"]["max_side"]
    names, images, grays = [], [], []
    for name, bgr in load_image_set(folder):
        scale = max_side / max(bgr.shape[:2])
        if scale < 1:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        names.append(name)
        images.append(bgr)
        grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return names, images, grays

def match_adjacent(method: str, features, cfg: dict) -> list:
    # Matches putativos de cada par de imagens adjacentes: a posição i é o par (i, i+1) 
    return [match_features(method, features[i], features[i + 1], cfg)[0] for i in range(len(features) - 1)]

def estimate_adjacent(features, matches, cfg: dict) -> list:
    pairs = []
    for i, pair_matches in enumerate(matches):
        H, inliers, error = estimate_homography(features[i].kps, features[i + 1].kps, pair_matches, cfg)
        n_inliers = int(inliers.sum()) if H is not None else 0
        pairs.append({"H": H, "inliers": inliers, "N_matches": len(pair_matches), "N_inliers": n_inliers,
                      "inlier_ratio": n_inliers / len(pair_matches) if pair_matches else 0.0,
                      "reproj_err_px": error})
    return pairs 