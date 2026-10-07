"""Keypoint detection utilities."""

import cv2
import time
import numpy as np

def compute_keypoints(image, detector_type):
    """Compute keypoints for the given image using the specified detector.

    Args:
        image (numpy.ndarray): Input image.
        detector_type (str): Type of keypoint detector to use ('SIFT', 'SURF', 'ORB', etc.).
    Returns:
        list: List of cv2.KeyPoint objects.
    """
    if detector_type == 'SIFT':
        sift = cv2.SIFT_create()
        t0 = time.time()
        keypoints = sift.detect(image, None)
        keypoints = sorted(keypoints, key=lambda x: -x.response)  # Sort by response (strongest first)
        t1 = time.time()
        sift_detection_time = t1 - t0
        return keypoints, {'detection_time': sift_detection_time}
    elif detector_type == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        t0 = time.time()
        keypoints = surf.detect(image, None)
        keypoints = sorted(keypoints, key=lambda x: -x.response)  # Sort by response (strongest first)
        t1 = time.time()
        surf_detection_time = t1 - t0
        return keypoints, {'detection_time': surf_detection_time}
    elif detector_type == 'ORB':
        orb = cv2.ORB_create()
        t0 = time.time()
        keypoints = orb.detect(image, None)
        keypoints = sorted(keypoints, key=lambda x: -x.response)  # Sort by response (strongest first)
        t1 = time.time()
        orb_detection_time = t1 - t0
        return keypoints, {'detection_time': orb_detection_time}
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")