# Teste sintético 

import cv2
import numpy as np
from datasets import ImagePair, load_image

def circular_mask(gray, feather_px: float, fill: int): 
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    radius = min(h, w) / 2 - 1
    weight = np.clip((radius - dist) / feather_px, 0.0, 1.0)   # 1 dentro, 0 fora
    return np.round(gray * weight + fill * (1 - weight)).astype(np.uint8)
 
def rotate(gray, angle_deg: float, fill: int): 
    h, w = gray.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return rotated, np.vstack([M, [0, 0, 1]])

def scale(gray, s: float, fill: int):  
    h, w = gray.shape
    new_w, new_h = round(w * s), round(h * s)
    # INTER_AREA ao reduzir evita aliasing | INTER_LINEAR ao ampliar
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
  
    left, top = (w - new_w) // 2, (h - new_h) // 2
    canvas = np.full((h, w), fill, dtype=np.uint8)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + new_w, w), min(top + new_h, h)
    canvas[y0:y1, x0:x1] = resized[y0 - top:y1 - top, x0 - left:x1 - left]

    # o cv2.resize trata os píxeis como centrados 
    sx, sy = new_w / w, new_h / h
    H = np.array([[sx, 0, 0.5 * sx - 0.5 + left],
                  [0, sy, 0.5 * sy - 0.5 + top],
                  [0, 0, 1]])
    return canvas, H

def make_pairs(cfg: dict) -> list[ImagePair]:
    # pares sintéticos 
    params = cfg["synthetic"]
    _, gray = load_image(params["base_image"])
    fill = int(round(gray.mean()))   # o mesmo cinzento de fundo em todas as operações

    pairs = []
    masked = circular_mask(gray, params["mask_feather_px"], fill)
    for angle in params["rotations_deg"]:
        rotated, H = rotate(masked, angle, fill)
        pairs.append(ImagePair("graf_rot", "rot", float(angle), masked, rotated, H, control=(angle == 0)))
    for s in params["scales"]:
        scaled, H = scale(gray, s, fill)
        pairs.append(ImagePair("graf_scale", "scale", float(s), gray, scaled, H, control=(s == 1.0)))
    return pairs