"""Image encoders for converting local descriptors into global features.

Contents:
    - Spatial tiling and histogram post-processing utilities.
    - Loading of saved encoder configurations from NumPy archives.
    - PCA projection and K-Means prediction wrappers.
    - Bag-of-Visual-Words, VLAD, and Fisher vector encoders.
    - Image-level feature extraction, caching, tiling, and aggregation.

The encoders use saved vocabularies or Gaussian mixture parameters and can
optionally apply a learned PCA projection before aggregation.
"""

import numpy as np, math
import os, os.path as osp
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.feature import learn_gmm, fisher_vector
from sklearn.mixture import GaussianMixture

from cyvlfeat.vlad import vlad
from cyvlfeat.fisher import fisher as vl_fisher
from cyvlfeat.kmeans import kmeans as vl_kmeans
from .features import computeFeatures, standarizeImage, extractFeatures

################################################################################

def spatial_tiling(
    kpts: np.ndarray,
    subdiv: np.ndarray,
    hw: Tuple[int, int],
    verbose: bool = False,
) -> Iterator[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    """Yield a keypoint mask and pixel bounds for each normalized tile.

    Args:
        kpts: Keypoint coordinates with rows in ``(y, x)`` order.
        subdiv: Normalized tile bounds as ``(min_y, min_x, max_y, max_x)``.
        hw: Image height and width in pixels.
        verbose: Whether to print the bounds and keypoint count for each tile.

    Yields:
        A boolean mask selecting keypoints in the tile and its pixel bounds.
    """
    fy, fx = (kpts[:, 0], kpts[:, 1])
    h, w = hw
    for i in range(subdiv.shape[0]):
        miny, minx, maxy, maxx = subdiv[i,:].astype(float) * np.array([h, w, h, w]).astype(float)
        ok_x = (fx >= minx) & (fx < maxx)
        ok_y = (fy >= miny) & (fy < maxy)
        ok =  ok_x & ok_y
        if verbose: print(f"Subdivision {i}: ({minx}, {miny}) to ({maxx}, {maxy}), Number of keypoints: {np.sum(ok)}")
        yield ok, (minx, miny, maxx, maxy)

def removeSpatialInformation(histograms: np.ndarray) -> np.ndarray:
    """Average four spatially tiled histograms into one histogram per image.

    Args:
        histograms: Array of shape ``(N, num_words * 4)``.

    Returns:
        Array of shape ``(N, num_words)`` with the tile dimension averaged.
    """
    N = histograms.shape[0]
    return np.mean(np.reshape(histograms, (N, -1, 4)), axis=2)

def load_npz(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load an encoder configuration from an ``.npz`` file.

    Args:
        file_path: Path to the NumPy archive containing encoder parameters.

    Returns:
        A dictionary mapping each archive key to its stored NumPy array.
    """
    with np.load(file_path, allow_pickle=True) as data:
        encoder_cfg = {key: data[key] for key in data.files}
    return encoder_cfg


class PCAProjector:
    """Project descriptors from ``D`` dimensions to ``C`` dimensions.

    ``D`` is the original descriptor dimension and ``C`` is the configured
    value of ``pca_dim``. When PCA is disabled, the input shape is preserved.
    """
    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialize a projector from a saved encoder configuration.

        Args:
            cfg: Configuration containing ``pca_dim``, ``projection``, and
                optionally ``projection_mean``.
        """
        if cfg.get("pca_dim", 0) > 0:
            self.pca = PCA(n_components=cfg["pca_dim"], random_state=0)
            self.pca.components_ = cfg["projection"]
            self.pca.mean_ = cfg.get("projection_mean", np.zeros(cfg["projection"].shape[1]))
        else:
            self.pca = None

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Apply the saved PCA projection to a descriptor matrix.

        Args:
            descriptors: Array of shape ``(num_descriptors, descriptor_dim)``.

        Returns:
            The projected descriptor array, or the original array when PCA is
            disabled.
        """
        if self.pca is None:
            return descriptors  # No projection applied.
        else:
            # Center the descriptors and apply the PCA transformation.
            centered = descriptors - self.pca.mean_
            proj_ =  centered @ self.pca.components_.T
            return proj_


class KMeansPredictor:
    """Assign ``N`` descriptor rows to one of ``K`` visual-word centers.

    Cluster centers have shape ``(K, D)``. Predictions have shape ``(N,)`` and
    distances returned by :meth:`predict_transform` also have shape ``(N,)``.
    """
    def __init__(self, centers: np.ndarray) -> None:
        """Initialize a predictor with the supplied cluster centers.

        Args:
            centers: Array of shape ``(num_clusters, descriptor_dim)``.
        """
        n_clusters = centers.shape[0]
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        self.clusterer.cluster_centers_ = centers.astype(np.float32)
        self.clusterer.n_features_in_ = centers.shape[1]
        self.clusterer._n_threads = 1

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        """Return the nearest cluster index for each descriptor.

        Args:
            descriptors: Array of descriptor rows to classify.

        Returns:
            Integer array containing one cluster index per descriptor.
        """
        return self.clusterer.predict(descriptors.astype(np.float32))

    def predict_transform(self, descriptors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return nearest cluster indices and distances for each descriptor.

        Args:
            descriptors: Array of descriptor rows to classify.

        Returns:
            A tuple containing integer cluster indices and corresponding
            distances to the assigned centers.
        """
        # get distances to all centers
        all_distances = self.clusterer.transform(descriptors.astype(np.float32))
        # return the index of the closest center for each descriptor
        assigned_indices = np.argmin(all_distances, axis=1)
        assigned_distances = all_distances[np.arange(len(assigned_indices)), assigned_indices]
        return assigned_indices, assigned_distances

#-------------------------------------------------------------------------------
# Encoding strategies for aggregating local descriptors into a global image representation.
#-------------------------------------------------------------------------------

class BoVWEncoder:
    """Encode ``(N, D)`` descriptors as a ``(K,)`` word histogram.

    ``K`` is ``cfg["num_words"]``. If PCA is enabled, the descriptors are
    projected from ``D`` dimensions to ``C`` before visual-word assignment.
    With ``with_dist=True``, the method returns a tuple containing the
    ``(K,)`` histogram, ``(N,)`` assignments, and ``(N,)`` distances.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialize the visual vocabulary and optional PCA projection.

        Args:
            cfg: Configuration containing ``words`` and ``num_words`` plus any
                optional PCA parameters.
        """
        self.cfg = cfg
        self.clusterer = KMeansPredictor(cfg["words"])
        self.proj = PCAProjector(cfg)

    def encode(
        self, descriptors: np.ndarray, with_dist: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Encode descriptors as a histogram of visual-word assignments.

        Args:
            descriptors: Array of local descriptor rows.
            with_dist: If true, also return assignments and assignment
                distances.

        Returns:
            A ``(K,)`` histogram. When ``with_dist`` is true, returns
            ``(histogram, assignments, distances)`` with shapes ``(K,)``,
            ``(N,)``, and ``(N,)`` respectively.
        """
        # Apply PCA projection when it was learned during encoder training.
        descriptors = self.proj.transform(descriptors)
        # print(descriptors.shape)
        if with_dist:
            indices, distances = self.clusterer.predict_transform(descriptors)
            # Count how many descriptors fall into each visual-word bin.
            bovw_hist, _ = np.histogram(indices, bins=np.arange(self.cfg["num_words"] + 1))
            return bovw_hist, indices, distances
        else:
            indices = self.clusterer.predict(descriptors.astype(np.float32))
            # Count how many descriptors fall into each visual-word bin.
            bovw_hist, _ = np.histogram(indices,  bins=np.arange(self.cfg["num_words"] + 1))
        return bovw_hist


class VLADEncoder:
    """Encode ``(N, D)`` descriptors as a flattened VLAD vector.

    With ``K`` visual words and projected descriptor dimension ``C``, the
    output has shape ``(K * C,)``. Without PCA, ``C`` equals ``D``.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialize the visual vocabulary and optional PCA projection.

        Args:
            cfg: Configuration containing the VLAD centers and word count,
                plus any optional PCA parameters.
        """
        self.cfg = cfg
        self.clusterer = KMeansPredictor(cfg["words"])
        self.proj = PCAProjector(cfg)

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """Encode descriptors as a normalized VLAD residual vector.

        Args:
            descriptors: Array of local descriptor rows.

        Returns:
            A one-dimensional VLAD feature vector with shape ``(K * C,)``.
        """
        # Reduce descriptor dimensionality before estimating the Fisher vector.
        descriptors = self.proj.transform(descriptors)
        # Use the cyvlfeat library for efficient VLAD encoding.
        words = self.clusterer.predict(descriptors.astype(np.float32))
        # Create a one-hot encoding of the cluster assignments for each descriptor.
        assignments = np.zeros((words.shape[0], self.cfg["num_words"]), dtype=np.float32)
        assignments[np.arange(words.shape[0]), words] = 1.0
        # Compute the VLAD vector using the cyvlfeat library, which efficiently
        # accumulates residuals and applies normalization.
        vlad_vector = vlad(
            descriptors.astype(np.float32), 
            self.cfg["words"].astype(np.float32),
            assignments,
            normalize_components=False,  # Disable per-cluster normalization 
            normalize_mass=False,        # Disable mass normalization
            square_root=False,           # Disable signed square root
            unnormalized=False           # Apply overall normalization
        )
        vlad_vector = vlad_vector.flatten()
        return vlad_vector


class FisherVectorEncoder:
    """Encode ``(N, D)`` descriptors as a flattened Fisher vector.

    With ``G`` Gaussian components and projected descriptor dimension ``C``,
    the first- and second-order statistics produce an output of shape
    ``(2 * G * C,)``. Without PCA, ``C`` equals ``D``.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialize a Gaussian mixture model from saved parameters.

        Args:
            cfg: Configuration containing GMM ``means``, ``covariances``, and
                ``priors``, plus any optional PCA parameters.
        """
        self.cfg = cfg
        self.proj =PCAProjector(cfg)
        self.gmm = GaussianMixture(
            n_components=cfg["means"].shape[0], 
            covariance_type='diag', 
            random_state=0
        )

        # Manually inject the parameters
        self.gmm.means_ =  cfg["means"]
        self.gmm.covariances_ = cfg["covariances"]
        self.gmm.weights_ =  cfg["priors"]

        # CRITICAL STEP: Compute and set the Cholesky precision manually
        # For 'diag' covariance, this is just 1 / sqrt(covariances)
        self.gmm.precisions_cholesky_ = 1.0 / np.sqrt(cfg["covariances"])
        self.gmm.n_features_in_ = cfg["means"].shape[1]

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """Encode descriptors as a Fisher vector from the configured GMM.

        Args:
            descriptors: Array of local descriptor rows.

        Returns:
            A one-dimensional Fisher feature vector with shape
            ``(2 * G * C,)``.
        """
        # Reduce descriptor dimensionality before estimating the Fisher vector.
        descriptors = self.proj.transform(descriptors)
        # The Fisher vector captures first- and second-order statistics relative to
        # the learned Gaussian mixture model.
        return fisher_vector(descriptors, self.gmm)


class ImageEncoder:
    """Extract descriptors and concatenate encodings from up to four tiles.

    The extracted keypoints have shape ``(N, 2)`` and descriptors have shape
    ``(N, D)``. For a base encoder output dimension ``F``, the tiled result
    normally has shape ``(4 * F,)``; empty tiles are skipped, so the actual
    length can be smaller when a tile contains no keypoints.
    """

    def __init__(self, encoder_cfg: Union[str, Path, Dict[str, Any]]) -> None:
        """Initialize an image encoder from a configuration or NPZ path.

        Args:
            encoder_cfg: Encoder configuration dictionary or path to its NPZ
                archive. The configuration must specify ``type``.
        """
        if isinstance(encoder_cfg, (str, Path)):
            self.name = str(encoder_cfg).split("encoder_")[-1].split(".npz")[0]
            encoder_cfg = load_npz(encoder_cfg)
            
            # print("Loaded encoder configuration from:", encoder_cfg)
        self.encoder = self.setup_encoder(encoder_cfg)
        # self.subdivisions = encoder_cfg.get("subdivisions",None).astype(np.float32)
        self.subdivisions =  np.array([
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0, 1.0]
        ], dtype=np.float32)
        # print(f"Feature extraction function set to: {self.ffunc}")
        self.type = encoder_cfg.get("type", "unknown")

    @staticmethod
    def setup_encoder(encoder: Dict[str, Any]) -> Union[BoVWEncoder, VLADEncoder, FisherVectorEncoder]:
        """Construct the concrete encoder selected by the configuration type.

        Args:
            encoder: Configuration dictionary whose ``type`` is ``bovw``,
                ``vlad``, or ``fv``.

        Returns:
            An encoder instance matching the requested type.

        Raises:
            ValueError: If the configuration specifies an unknown type.
        """
        if encoder["type"] == "bovw":
            return BoVWEncoder(encoder)
        elif encoder["type"] == "vlad":
            return VLADEncoder(encoder)
        elif encoder["type"] == "fv":
            return FisherVectorEncoder(encoder)
        else:
            raise ValueError(f"Unknown encoder type: {encoder['type']}")

    def computeFeatures(
        self,
        image: Any,
        cache_dir: Union[str, Path, None] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load cached features or extract keypoints and descriptors.

        Args:
            image: Image path or image object accepted by ``computeFeatures``.
            cache_dir: Optional directory containing cached feature archives.
            verbose: Whether to request verbose feature extraction.

        Returns:
            A tuple ``(keypoints, descriptors, image_size)``.
        """
        cache_loaded = False
        if isinstance(image, (str, Path)) and cache_dir is not None:
            image = str(image)
            # Check if the encoded features are already cached.
            cache_file = osp.join(cache_dir, f"{Path(image).stem}.npz")
            if verbose: print("Checking for cached features at:", cache_file)
            if osp.exists(cache_file):
                data = np.load(cache_file)
                keypoints = data['keypoints']
                descriptors = data['descriptors']
                image_size = data['image_size']
                if verbose: print(f"Loaded cached features for {image} from {cache_file}")
                cache_loaded = True
        if not cache_loaded:
            # Normalize the image and extract local descriptors with the configured
            # feature extractor. The step size controls the spatial stride of the
            # dense extraction pipeline.
            keypoints, descriptors, image_size = computeFeatures(image)
        return keypoints, descriptors, image_size
    
    def encode_tiling(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        image_size: np.ndarray,
    ) -> np.ndarray:
        """Encode descriptors independently within each spatial tile.

        Args:
            keypoints: Keypoint coordinates in ``(y, x)`` order.
            descriptors: Descriptor row for each keypoint.
            image_size: Image dimensions used to convert normalized tile bounds
                into pixels.

        Returns:
            A one-dimensional vector formed by concatenating the non-empty
            tile encodings. Its maximum shape is ``(4 * F,)``.
        """
        # Split the image into spatial tiles and encode descriptors per region.
        spt_feat = []
        # print(self.subdivisions,image_size[:2])
        spt = spatial_tiling(keypoints, self.subdivisions, image_size[:2], 
                                verbose=False)
        for ok, (x1, y1, x2, y2) in spt:
            if sum(ok) == 0:
                print("No keypoints in subdivision:", (x1, y1, x2, y2))
                continue  # Skip if no keypoints in this subdivision.
            z = self.encoder.encode(descriptors[ok,:])
            spt_feat.append(z)
        spt_feat = np.concatenate(spt_feat)
        return spt_feat

    def get_enconding(
        self,
        kpts: np.ndarray,
        desc: np.ndarray,
        isize: np.ndarray,
        tiling: bool = False,
    ) -> np.ndarray:
        """Return a global encoding or a concatenated tiled encoding.

        Args:
            kpts: Keypoint coordinates corresponding to ``desc``.
            desc: Local descriptor rows.
            isize: Image dimensions used for spatial tiling.
            tiling: Whether to encode each spatial tile separately.

        Returns:
            The resulting global or tiled feature vector.
        """
        if tiling:
            return self.encode_tiling(kpts, desc, isize)
        else:
            return self.encoder.encode(desc)

    def get_visual_words(
        self, image: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return visual-word counts, assignments, distances, and keypoints.

        Args:
            image: Image path or image object accepted by feature extraction.

        Returns:
            A tuple ``(histogram, assignments, distances, keypoints)``.

        Raises:
            AssertionError: If the configured encoder is not BoVW.
        """
        assert self.type in ["bovw"], "Visual words are only available for BoVW and VLAD encoders."
        # Project descriptors if PCA was applied during training.
        kpts, descs, isize = self.computeFeatures(image)
        print("Number of keypoints:", kpts.shape[0], "Descriptor dimension:", descs.shape[1])
        hist, words, distances  = self.encoder.encode(descs, with_dist=True)
        return hist, words, distances, kpts

    def encode(
        self,
        image: Any,
        cache_dir: Union[str, Path, None] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Extract or load image features and return their tiled encoding.

        Args:
            image: Image path or image object accepted by feature extraction.
            cache_dir: Optional directory containing cached feature archives.
            verbose: Whether to request verbose feature extraction.

        Returns:
            The concatenated encoding for the image's spatial tiles.
        """
        # If a cache directory is supplied and an image has already been processed,
        # we can skip descriptor extraction and reuse the stored result.
        kpts, desc, isize = self.computeFeatures(image, cache_dir, verbose)
        return self.encode_tiling(kpts, desc, isize)


