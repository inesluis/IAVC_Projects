"""Estimação robusta de homografias (RANSAC), erro de reprojeção e composição 

Uma homografia H é uma matriz 3x3 que leva pontos de uma imagem para outra:
    [x', y', w']^T = H · [x, y, 1]^T    ->    ponto final = (x'/w', y'/w')

Convenção: a H devolvida leva pontos da imagem 1 para a imagem 2 (como a ground-truth do HPatches) 
"""

import cv2
import numpy as np

MIN_MATCHES = 4   # uma homografia tem 8 graus de liberdade e cada par de pontos dá 2 equações
 
def project_points(points, H: np.ndarray) -> np.ndarray:
    # Aplica a homografia H a pontos (N, 2) 
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, H).reshape(-1, 2)
 
def reprojection_errors(p1, p2, H: np.ndarray) -> np.ndarray:
    # Erro de reprojeção de cada par em píxeis: distância entre H · p1 e p2 
    return np.linalg.norm(project_points(p1, H) - p2, axis=1)
  
def estimate_homography(kps1, kps2, matches, cfg: dict):
    # Estima com RANSAC a homografia que leva a imagem 1 para a imagem 2

    """ RANSAC (dentro do cv2.findHomography):
      1. escolhe 4 matches ao acaso e calcula uma H candidata;
      2. conta os matches com erro de reprojeção abaixo do limiar (os inliers);
      3. repete e fica com a H que tem mais inliers;
      4. no fim, recalcula H com todos os inliers dessa hipótese.
    Um limiar pequeno dá poucos inliers; um limiar grande aceita matches errados e a H fica pior.

    Os parâmetros vêm de cfg["ransac"]. Devolve (H, máscara booleana dos inliers, erro médio de
    reprojeção dos inliers em px). Com menos de 4 matches ou se a estimação falhar: (None, None, nan).
    """
    if len(matches) < MIN_MATCHES:
        return None, None, np.nan
    p1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    ransac = cfg["ransac"]
    cv2.setRNGSeed(cfg["seed"])   # o RANSAC é aleatório: assim o resultado não depende das chamadas anteriores
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, ransacReprojThreshold=ransac["thr_px"],
                                 maxIters=ransac["max_iters"], confidence=ransac["confidence"])
    if H is None:
        return None, None, np.nan

    inliers = mask.ravel().astype(bool)
    if not inliers.any():
        return None, None, np.nan
    return H, inliers, float(reprojection_errors(p1[inliers], p2[inliers], H).mean())
 
def chain_to_reference(adjacent: list, ref_index: int) -> list:
    """Compõe homografias entre imagens adjacentes para as levar todas à mesma referência

    adjacent[i] leva a imagem i para a imagem i+1 (N imagens, N-1 matrizes). Devolve N matrizes:
    a posição i leva a imagem i para a imagem ref_index.

    Exemplo com 3 imagens A, B, C (adjacent = [H_AB, H_BC]):
        referência B (central):   A: H_AB    B: I             C: inv(H_BC)
        referência A (extremo):   A: I       B: inv(H_AB)     C: inv(H_AB) · inv(H_BC)

    Cada multiplicação junta o erro de mais uma homografia, por isso as imagens mais longe da
    referência ficam com mais erro acumulado.
    """
    n_images = len(adjacent) + 1
    to_ref = [None] * n_images
    to_ref[ref_index] = np.eye(3)

    for i in range(ref_index - 1, -1, -1): # à esquerda da referência: i -> i+1 -> ... -> ref
        to_ref[i] = to_ref[i + 1] @ adjacent[i]
    for i in range(ref_index + 1, n_images): # à direita: i -> i-1 -> ... -> ref (inversas)
        to_ref[i] = to_ref[i - 1] @ np.linalg.inv(adjacent[i - 1])
    return to_ref