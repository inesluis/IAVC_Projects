import numpy as np
import cv2

def compose_imgs(img1: np.ndarray, img2: np.ndarray, H: np.ndarray):
    """
    Compose 2 images given the transformation warping matrix of the second.

    :param img1: Image kept untransformed (reference)
    :param img2: Image to be warped
    :param H: 3x3 homography / projective transformation matrix
    :return: (imgcomposed, displacement)
             - imgcomposed: The blended composite image
             - displacement: [ty, tx] translation applied to img1
    """
    # image shape is (height, width) or (height, width, channels)
    sizey, sizex = img1.shape[:2]

    # Corners of img2/img1 in homogeneous coordinates: [x, y, 1]
    corners = np.array(
        [[1, 1, 1], [1, sizey, 1], [sizex, sizey, 1], [sizex, 1, 1]],
        dtype=np.float64,
    )

    # Project corners through H: cornersh = (H @ corners.T).T
    projected = (H @ corners.T).T
    cornersh = projected[:, :2] / projected[:, 2:3]

    minvals = np.min(cornersh, axis=0)
    maxvals = np.max(cornersh, axis=0)

    # Determine shift/translation if warped image spills left or top
    tx = int(np.floor(1 - minvals[0])) if minvals[0] < 1 else 0
    ty = int(np.floor(1 - minvals[1])) if minvals[1] < 1 else 0

    deltax = int(np.ceil(maxvals[0] - sizex)) if maxvals[0] > sizex else 0
    deltay = int(np.ceil(maxvals[1] - sizey)) if maxvals[1] > sizey else 0

    new_h = sizey + ty + deltay + 1
    new_w = sizex + tx + deltax

    # Translation matrix to shift origin by (tx, ty)
    T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])

    # Combined homography that warps and applies translation shift
    H_translated = T @ H

    # Warp img2 into the expanded canvas
    # Note: cv2.warpPerspective expects dsize=(width, height)
    img2_warped = cv2.warpPerspective(img2, H_translated, (new_w, new_h))

    # Place img1 shifted by (tx, ty) onto its canvas
    if img1.ndim == 3:
        newimg1 = np.zeros((new_h, new_w, img1.shape[2]), dtype=img1.dtype)
    else:
        newimg1 = np.zeros((new_h, new_w), dtype=img1.dtype)

    newimg1[ty : ty + sizey, tx : tx + sizex] = img1

    # Compose images using pixel-wise maximum
    imgcomposed = np.maximum(newimg1, img2_warped)
    displacement = [ty, tx]

    return imgcomposed, displacement
