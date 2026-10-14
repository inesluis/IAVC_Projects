"""
stitching.py — Construção de panoramas (warp + canvas + média)

Passos 9 a 12 do enunciado (§3.1):
     9. Deformar (warp) cada imagem para o referencial da imagem de referência.
    10. Criar um canvas suficientemente grande para conter todas as imagens.
    11. Compor as imagens; na zona de sobreposição faz-se a MÉDIA dos pixels.
    12. Repetir para várias imagens (ex.: 3 imagens, com referências diferentes).

Não se usa o cv2.Stitcher: o canvas e a composição são feitos aqui.

Funções:
    load_panorama_images()      -> lê um conjunto de imagens (reduz as muito grandes)
    match_adjacent()            -> matches de cada par de imagens adjacentes
    estimate_adjacent()         -> homografia RANSAC e estatísticas de cada par
    compute_canvas()            -> tamanho do canvas e translação T
    warp_to_canvas()            -> deforma uma imagem para o canvas + máscara
    blend_average()             -> média das imagens na sobreposição
    stitch_images()             -> panorama de N imagens (dadas as homografias)
    stitch_pair()               -> atalho para 2 imagens

Teste rápido (2 ou mais imagens, da esquerda para a direita):
    python src/stitching.py data/Panorama/keble_a.jpg data/Panorama/keble_b.jpg
    python src/stitching.py data/Panorama/keble_a.jpg data/Panorama/keble_b.jpg data/Panorama/keble_c.jpg --ref 1
    python src/stitching.py data/Panorama/keble_a.jpg data/Panorama/keble_b.jpg --method SuperPoint+LightGlue
"""

import argparse

import cv2
import numpy as np

from config import get_config, set_seeds
from datasets import imwrite, load_image, load_image_set
from homography import chain_to_reference, estimate_homography, project_points
from methods import ALL_METHODS, extract_features, match_features

# Limite de segurança: uma homografia errada pode projetar os cantos para muito
# longe e pedir um canvas gigantesco (que esgota a memória).
MAX_CANVAS_SIDE = 20000  # px


# ---------------------------------------------------------------------------
# CANVAS
# ---------------------------------------------------------------------------
def image_corners(image: np.ndarray) -> np.ndarray:
    """Os 4 cantos de uma imagem, como array (4, 2) de pontos (x, y)."""
    h, w = image.shape[:2]
    return np.float64([[0, 0], [w, 0], [w, h], [0, h]])


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
    """
    Constrói o panorama de N imagens a cores.

    images:        lista de imagens BGR
    homographies:  homographies[i] leva a imagem i para a imagem de REFERÊNCIA
                   (para a própria referência é a identidade, np.eye(3))
    """
    T, canvas_size = compute_canvas(images, homographies)

    warped_images, masks = [], []
    for image, H in zip(images, homographies):
        warped, mask = warp_to_canvas(image, H, T, canvas_size)
        warped_images.append(warped)
        masks.append(mask)

    return blend_average(warped_images, masks)


def stitch_pair(img_ref, img_mov, H_mov_to_ref):
    """Panorama de 2 imagens: img_mov é deformada para o referencial de img_ref."""
    return stitch_images([img_ref, img_mov], [np.eye(3), H_mov_to_ref])


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
    """Matches putativos de cada par de imagens adjacentes: a posição i é o par (i, i+1)."""
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


# ---------------------------------------------------------------------------
# TESTE RÁPIDO (só para validar o código)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_config()
    set_seeds(cfg)

    ap = argparse.ArgumentParser(description="Teste rápido do stitching.")
    ap.add_argument("images", nargs="+", help="2 ou mais imagens, por ordem (esquerda -> direita)")
    ap.add_argument("--method", default="SIFT", choices=ALL_METHODS, help="detetor/descritor/matcher")
    ap.add_argument("--ref", type=int, default=None, help="índice da imagem de referência (por omissão: a central)")
    ap.add_argument("--thr", type=float, default=None, help="limiar do RANSAC em px (por omissão: o do config)")
    ap.add_argument("--out", default=str(cfg["paths"]["panoramas"] / "panorama.jpg"))
    args = ap.parse_args()

    if len(args.images) < 2:
        ap.error("são precisas pelo menos 2 imagens")
    if args.thr is not None:
        cfg["ransac"]["thr_px"] = args.thr

    loaded = [load_image(p) for p in args.images] # (cores, cinzento); o cinzento é só para detetar
    images = [bgr for bgr, _ in loaded]
    features = [extract_features(args.method, gray, cfg) for _, gray in loaded]
    ref = len(images) // 2 if args.ref is None else args.ref

    # Homografia entre cada par de imagens adjacentes: i -> i+1
    pairs = estimate_adjacent(features, match_adjacent(args.method, features, cfg), cfg)
    for i, pair in enumerate(pairs):
        if pair["H"] is None:
            raise SystemExit(f"Não foi possível estimar a homografia do par {i}->{i + 1} "
                             f"({pair['N_matches']} matches).")
        print(f"par {i}->{i + 1}: matches={pair['N_matches']} inliers={pair['N_inliers']} "
              f"rácio={pair['inlier_ratio']:.3f} erro médio={pair['reproj_err_px']:.3f} px")

    # Leva todas as imagens para a referência e compõe o panorama
    panorama = stitch_images(images, chain_to_reference([pair["H"] for pair in pairs], ref))
    imwrite(args.out, panorama)
    print(f"método={args.method} referência={ref} panorama={panorama.shape[1]}x{panorama.shape[0]} px -> {args.out}")
