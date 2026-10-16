# stitching.py — Construção de panoramas (warp + canvas + média)

import cv2
import numpy as np
from datasets import load_image_set
from homography import chain_to_reference, estimate_homography, image_corners, project_points
from methods import match_features

# Limite de segurança: uma homografia errada pode projetar os cantos para muito
# longe e pedir um canvas gigantesco
MAX_CANVAS_SIDE = 20000  # px

# CANVAS
def compute_canvas(images, homographies):
    """
    Calcula o tamanho do canvas que contém todas as imagens depois de deformadas.

    Projeta os 4 cantos de cada imagem com a sua homografia e fica com o
    retângulo que os contém a todos (mínimos e máximos de x e de y).

    Os cantos podem cair em coordenadas NEGATIVAS (à esquerda ou acima da
    referência), e uma imagem não tem pixels negativos. Por isso devolve-se
    também uma translação T que empurra tudo para coordenadas positivas.

    Devolve (T, (largura, altura)).
    """
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

# ---------------------------------------------------------------------------
# WARP + COMPOSIÇÃO
# ---------------------------------------------------------------------------
def warp_to_canvas(image, H, T, canvas_size):
    """
    Deforma uma imagem para o canvas. Devolve (imagem deformada, máscara).

    A transformação total é T @ H: primeiro H leva a imagem para o referencial
    da referência, depois T desloca-a para dentro do canvas.

    A máscara diz que pixels do canvas pertencem a esta imagem (1 = pertence,
    0 = vazio). É obtida deformando uma imagem só com 1s, e não testando
    "pixel > 0": assim os pixels pretos verdadeiros da imagem não se perdem.
    """
    M = T @ H
    warped = cv2.warpPerspective(image, M, canvas_size).astype(np.float32)

    ones = np.ones(image.shape[:2], dtype=np.float32)
    mask = cv2.warpPerspective(ones, M, canvas_size)
    return warped, mask

def blend_average(warped_images, masks):
    """
    Junta as imagens deformadas, fazendo a média onde se sobrepõem.

    Em cada pixel:  panorama = soma das imagens / número de imagens que lá existem
    (a soma das máscaras é exatamente esse número: 1 fora da sobreposição,
    2 onde duas imagens coincidem, etc.).
    """
    total = np.sum(warped_images, axis=0)      # (H, W, 3) soma das cores
    count = np.sum(masks, axis=0)[..., None]   # (H, W, 1) nº de imagens em cada pixel

    # Onde não há nenhuma imagem (count = 0) o panorama fica preto.
    panorama = np.divide(total, count, out=np.zeros_like(total), where=count > 1e-6)
    return np.clip(panorama, 0, 255).astype(np.uint8)

def stitch_images(images, homographies):
    # Constrói o panorama de N imagens a cores.

    # images:        lista de imagens BGR
    # homographies:  homographies[i] leva a imagem i para a imagem de REFERÊNCIA
    #                (para a própria referência é a identidade, np.eye(3))
     
    T, canvas_size = compute_canvas(images, homographies)

    warped_images, masks = [], []
    for image, H in zip(images, homographies):
        warped, mask = warp_to_canvas(image, H, T, canvas_size)
        warped_images.append(warped)
        masks.append(mask)

    return blend_average(warped_images, masks)

def stitch_or_none(images, pairs, ref: int):
    # Panorama com a imagem de referência dada, ou None se faltar uma homografia ou o canvas for absurdo
    if any(pair["H"] is None for pair in pairs):
        return None
    try:
        return stitch_images(images, chain_to_reference([pair["H"] for pair in pairs], ref))
    except ValueError as error:
        print(error)
        return None

# ---------------------------------------------------------------------------
# DAS IMAGENS ÀS HOMOGRAFIAS
# ---------------------------------------------------------------------------
def load_panorama_images(folder, cfg: dict):
    """
    Lê as imagens de um conjunto, por ordem de nome (esquerda -> direita).

    As fotografias com o lado maior acima de cfg["stitching"]["max_side"] são
    reduzidas: um panorama de fotografias de 12 Mpx não cabe na memória.

    Devolve (nomes, imagens BGR, imagens em cinzento). O cinzento é só para
    detetar; o panorama é feito com as imagens a cores.
    """
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
    """
    Homografia RANSAC de cada par de imagens adjacentes. Devolve um dicionário por par:
        H (leva a imagem i para a i+1, ou None se falhou), inliers (máscara),
        N_matches, N_inliers, inlier_ratio, reproj_err_px
    """
    pairs = []
    for i, pair_matches in enumerate(matches):
        H, inliers, error = estimate_homography(features[i].kps, features[i + 1].kps, pair_matches, cfg)
        n_inliers = int(inliers.sum()) if H is not None else 0
        pairs.append({"H": H, "inliers": inliers, "N_matches": len(pair_matches), "N_inliers": n_inliers,
                      "inlier_ratio": n_inliers / len(pair_matches) if pair_matches else 0.0,
                      "reproj_err_px": error})
    return pairs 