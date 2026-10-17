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

def image_corners(image: np.ndarray) -> np.ndarray:
    # Os 4 cantos de uma imagem, como array (4, 2) de pontos (x, y)
    h, w = image.shape[:2]
    return np.float64([[0, 0], [w, 0], [w, h], [0, h]])

def corner_distance(H_a, H_b, image: np.ndarray) -> float:
    # Distância média (px) entre os 4 cantos da imagem projetados com H_a e com H_b.
    # 0 = mesma geometria. nan se alguma das homografias não existe.
    if H_a is None or H_b is None:
        return np.nan
    corners = image_corners(image)
    return float(np.linalg.norm(project_points(corners, H_a) - project_points(corners, H_b), axis=1).mean())

def estimate_homography(kps1, kps2, matches, cfg: dict):
    # Estima com RANSAC a homografia que leva a imagem 1 para a imagem 2 
    if len(matches) < MIN_MATCHES:
        return None, None, np.nan
    p1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    ransac = cfg["ransac"]
    cv2.setRNGSeed(cfg["seed"])    
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, ransacReprojThreshold=ransac["thr_px"],
                                 maxIters=ransac["max_iters"], confidence=ransac["confidence"])
    if H is None:
        return None, None, np.nan

    inliers = mask.ravel().astype(bool)
    if not inliers.any():
        return None, None, np.nan
    return H, inliers, float(reprojection_errors(p1[inliers], p2[inliers], H).mean())
 
def chain_to_reference(adjacent: list, ref_index: int) -> list:
    n_images = len(adjacent) + 1
    to_ref = [None] * n_images
    to_ref[ref_index] = np.eye(3)

    for i in range(ref_index - 1, -1, -1): # à esquerda da referência: i -> i+1 -> ... -> ref
        to_ref[i] = to_ref[i + 1] @ adjacent[i]
    for i in range(ref_index + 1, n_images): # à direita: i -> i-1 -> ... -> ref (inversas)
        to_ref[i] = to_ref[i - 1] @ np.linalg.inv(adjacent[i - 1])
    return to_ref