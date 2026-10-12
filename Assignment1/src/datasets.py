"""Leitura dos dados: HPatches, GRAF e imagens para panoramas.  [Pessoa A]"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ImagePair:
    """Par de imagens com a homografia ground-truth H, que leva pontos da imagem 1 para a imagem k."""
    seq: str                 # nome da sequência, ex.: "v_graffiti"
    category: str            # "i" (iluminação), "v" (ponto de vista), "rot", "scale" ou "graf"
    k: float                 # nº da imagem k (HPatches, GRAF) ou nível (ângulo, fator de escala)
    gray1: np.ndarray        # imagens em cinzento (uint8), usadas para detetar
    grayk: np.ndarray
    H: np.ndarray
    bgr1: np.ndarray = None  # imagens a cores, só para as figuras
    bgrk: np.ndarray = None
    control: bool = False    # par trivial do teste sintético (0 graus, escala 1), fora das médias


def imread(path) -> np.ndarray:
    """Lê uma imagem a cores (BGR). Usa np.fromfile + imdecode porque o cv2.imread falha, sem
    aviso, em caminhos com acentos no Windows (D08)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    return img


def imwrite(path, img) -> None:
    """Grava uma imagem; pelo mesmo motivo do imread, usa imencode + tofile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, img)[1].tofile(str(path))


def load_image(path):
    """Lê uma imagem e devolve (cores BGR, cinzento)."""
    bgr = imread(path)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def list_sequences(root, categories, only=None) -> list[str]:
    """Nomes das sequências HPatches das categorias pedidas, por ordem alfabética.
    `only` restringe a algumas sequências (para testes rápidos)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta HPatches não encontrada: {root} (ver README, secção Dados)")
    seqs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name[0] in categories)
    if only is not None:
        seqs = [s for s in seqs if s in only]
    return seqs


def hpatches_pairs(cfg: dict, only=None):
    """Gera os pares (imagem 1, imagem k) do HPatches, sequência a sequência.
    Cada pasta tem 1.ppm ... 6.ppm e as homografias H_1_2 ... H_1_6."""
    root = Path(cfg["paths"]["hpatches"])
    for seq in list_sequences(root, cfg["eval"]["categories"], only):
        bgr1, gray1 = load_image(root / seq / "1.ppm")
        for k in cfg["eval"]["pairs_k"]:
            bgrk, grayk = load_image(root / seq / f"{k}.ppm")
            H = np.loadtxt(str(root / seq / f"H_1_{k}"))
            yield ImagePair(seq, seq[0], k, gray1, grayk, H, bgr1, bgrk)


# Só temos 3 imagens GRAF. No relatório chamam-se 1, 2 e 3; a imagem 3 é o ficheiro img4.ppm (D21).
GRAF_FILES = {2: "img2.ppm", 3: "img4.ppm"}
GRAF_HOMOGRAPHIES = {2: "H1to2p.txt", 3: "H1to4p.txt"}


def load_graf_pairs(cfg: dict) -> list[ImagePair]:
    """Os dois pares GRAF: imagem 1 -> imagem 2 e imagem 1 -> imagem 3."""
    root = Path(cfg["paths"]["graf"])
    bgr1, gray1 = load_image(root / "img1.ppm")
    pairs = []
    for k in (2, 3):
        bgrk, grayk = load_image(root / GRAF_FILES[k])
        H = np.loadtxt(str(root / GRAF_HOMOGRAPHIES[k]))
        pairs.append(ImagePair("graf", "graf", k, gray1, grayk, H, bgr1, bgrk))
    return pairs


def load_image_set(folder, pattern: str = "*.jpg"):
    """Lê as imagens de uma pasta por ordem de nome (para os panoramas). Devolve [(nome, BGR), ...]."""
    return [(p.name, imread(p)) for p in sorted(Path(folder).glob(pattern))]
