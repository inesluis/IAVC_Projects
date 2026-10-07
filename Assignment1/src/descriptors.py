"""Feature descriptor utilities."""

import cv2
import time
import numpy as np

def compute_descriptors(keypoints, image, descriptor_type):
    """Compute feature descriptors for the given keypoints in the image.

    Args:
        keypoints (list): List of cv2.KeyPoint objects.
        image (numpy.ndarray): Input image.
        descriptor_type (str): Type of descriptor to compute ('SIFT', 'SURF', 'ORB', etc.).

    Returns:
        numpy.ndarray: Computed descriptors.
    """
    if descriptor_type == 'SIFT':
        sift = cv2.SIFT_create()
        t0 = time.time()
        _, descriptors = sift.compute(image, keypoints)
        t1 = time.time()
        sift__descriptor_time = t1 - t0
        return descriptors, {'descriptor_time': sift__descriptor_time}
    elif descriptor_type == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        t0 = time.time()
        _, descriptors = surf.compute(image, keypoints)
        t1 = time.time()
        surf__descriptor_time = t1 - t0
        return descriptors, {'descriptor_time': surf__descriptor_time}
    elif descriptor_type == 'ORB':
        orb = cv2.ORB_create()
        t0 = time.time()
        _, descriptors = orb.compute(image, keypoints)
        t1 = time.time()
        orb__descriptor_time = t1 - t0
        return descriptors, {'descriptor_time': orb__descriptor_time}
    else:
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")



