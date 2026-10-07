"""Leitura de dados: pares do HPatches (seq, categoria i/v, k, img1, imgk, H_1_k), GRAF e panoramas.  [Pessoa A]"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class ImagePair:
    """Par de imagens com homografia ground-truth H (leva pontos da img1 para a imgk).

    bgr1/bgrk: imagens a cores (BGR, uint8); gray1/grayk: cinzento (uint8), usadas para detetar.
    category: 'i' (iluminação), 'v' (ponto de vista) ou '' (fora do HPatches, ex.: GRAF).
    """
    seq: str
    category: str
    k: int
    bgr1: np.ndarray
    bgrk: np.ndarray
    gray1: np.ndarray
    grayk: np.ndarray
    H: np.ndarray


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread que funciona com caminhos com acentos no Windows (D08).
    Lança FileNotFoundError se o ficheiro não existir ou não for descodificável."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags) if data.size else None
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    """cv2.imwrite que funciona com caminhos com acentos no Windows (D08). Cria a pasta se preciso."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"Não foi possível codificar a imagem: {path}")
    buf.tofile(str(path))


def load_image(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê uma imagem e devolve (bgr, gray), ambas uint8."""
    bgr = imread(path, cv2.IMREAD_COLOR)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def load_homography(path: str | Path) -> np.ndarray:
    """Lê uma homografia 3x3 em texto (formato HPatches/Oxford) como float64."""
    H = np.loadtxt(str(path), dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homografia com forma inválida {H.shape}: {path}")
    return H


def list_sequences(root: str | Path, categories=("i", "v"), names=None) -> list[str]:
    """Lista (ordenada) as sequências HPatches de `root` cujas categorias estão em `categories`.
    `names` restringe a um subconjunto fixo (ex.: para testes rápidos)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta HPatches não encontrada: {root} (ver README, secção Dados)")
    seqs = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name.split("_", 1)[0] in categories)
    if names is not None:
        seqs = [s for s in seqs if s in set(names)]
    return seqs


def iter_hpatches_pairs(root: str | Path, categories=("i", "v"), seqs=None,
                        ks=(2, 3, 4, 5, 6)) -> Iterator[ImagePair]:
    """Itera sobre os pares (1 -> k) do HPatches, sequência a sequência, por ordem fixa.
    A imagem 1 é lida uma única vez por sequência."""
    root = Path(root)
    for seq in list_sequences(root, categories, seqs):
        d = root / seq
        bgr1, gray1 = load_image(d / "1.ppm")
        for k in ks:
            bgrk, grayk = load_image(d / f"{k}.ppm")
            yield ImagePair(seq=seq, category=seq.split("_", 1)[0], k=k,
                            bgr1=bgr1, bgrk=bgrk, gray1=gray1, grayk=grayk,
                            H=load_homography(d / f"H_1_{k}"))


def count_hpatches_pairs(root: str | Path, categories=("i", "v"), seqs=None, ks=(2, 3, 4, 5, 6)) -> int:
    """Número de pares que iter_hpatches_pairs vai produzir (útil para barras de progresso)."""
    return len(list_sequences(root, categories, seqs)) * len(ks)


def load_graf_pair(root: str | Path, k: int) -> ImagePair:
    """Par GRAF img1 -> imgk (k em {2, 4}) com H1to{k}p.txt."""
    root = Path(root)
    bgr1, gray1 = load_image(root / "img1.ppm")
    bgrk, grayk = load_image(root / f"img{k}.ppm")
    return ImagePair(seq="graf", category="", k=k, bgr1=bgr1, bgrk=bgrk,
                     gray1=gray1, grayk=grayk, H=load_homography(root / f"H1to{k}p.txt"))


def load_image_set(folder: str | Path, pattern: str = "*.jpg") -> list[tuple[str, np.ndarray]]:
    """Lê um conjunto de imagens para panorama, ordenado pelo nome. Devolve [(nome, bgr), ...]."""
    paths = sorted(Path(folder).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem '{pattern}' em {folder}")
    return [(p.name, imread(p)) for p in paths]
