import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def displayRankedImageList(scores, names, uniform=False, figzize=(8, 8)):
    """
    Display the ranked list of images based on the scores.
    """
    # Sort the scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    # create a 6 x 6 image grid
    grid_size = (6, 6)

    # Select images either from the beginning of the ranking or uniformly across it
    num_images = min(len(sorted_indices), grid_size[0] * grid_size[1])
    if uniform:
      selected_indices = np.linspace(
        0, len(sorted_indices) - 1, num_images, dtype=int
      )
    else:
      selected_indices = np.arange(num_images)
    
    # Create a figure to display the images
    plt.figure(figsize=figzize)
    plt.suptitle("Ranked Images", fontsize=20)
    # Loop through the top N images and display them
    for i, rank in enumerate(selected_indices):
        idx = sorted_indices[rank]
        img = Image.open(names[idx])  # Assuming load_image is a function that loads an image given its name
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.imshow(img)
        plt.title(f"Score: {scores[idx]:.2f}")
        plt.axis('off')
    plt.tight_layout()
    # set overall title

    plt.show()

def displayPrecisionRecallCurve(labels, scores, title="Precision-Recall Curve"):
    ap = average_precision_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall,
        average_precision=average_precision_score(labels, scores),
        name=f"LinearSVC"
    )

    fig, ax = plt.subplots()
    ax.set_title(title)
    pr_display.plot(ax=ax)
    plt.show()
    return precision, recall, thresholds, ap

class RelevantVisualWordsDisplay:

    def __init__(self, img, w, encoder):
        # compute features and get the visual words
        # img = standarizeImage(img_file)
        hist, words, distances, kpts = encoder.get_visual_words(img)
        hist = np.sqrt(hist) # Hellinger kernel
        hist = hist / (np.sum(hist**2) + 1e-10)**0.5 # L2 normalize
    
        # fold the w vetor, which may have been computed with spatial subdivisions
        w_ = w.reshape(hist.shape[0],-1)
        w_ = np.mean(w_, axis=1) # average the weights for each visual word
        weights = w_ * hist # get weights for each visual word

        self.img = img
        self.hist = hist
        self.kpts = kpts
        self.words = words
        self.distances = distances
          
        self.perm = np.argsort(weights)[::-1] # sort descending order
        self.scores = weights[self.perm]

    def build_mosaic(self, kpts, dists, imsize, ncols = None):
        dist_perm = np.argsort(dists)[::-1] # sort descending order
        uv = kpts[dist_perm,:2]
        s = np.round(kpts[dist_perm,3]*2)
        u0 = np.clip(uv[:,0]-s, 0, imsize[0]-1).astype(int)
        u1 = np.clip(uv[:,0]+s, 0, imsize[0]-1).astype(int)
        v0 = np.clip(uv[:,1]-s, 0, imsize[1]-1).astype(int)
        v1 = np.clip(uv[:,1]+s, 0, imsize[1]-1).astype(int)

        patches = []
        for i in range(u0.shape[0]):
            patch = self.img[u0[i]:u1[i],v0[i]:v1[i]]
            patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)
            patches.append(patch)

        if ncols is None:
            ncols = int(np.ceil(np.sqrt(len(patches))))
        
        # Combine all patches into one mosaic image
        nrows = int(np.ceil(len(patches) / ncols))
        h, w, c = patches[0].shape
        canvas = np.ones((nrows * h, ncols * w, c), dtype=patches[0].dtype)
        for idx, patch in enumerate(patches):
            row, col = divmod(idx, ncols)
            canvas[row*h:(row+1)*h, col*w:(col+1)*w] = patch

        return canvas
    
    def plot(self, rank, figsize=(8, 6)):
        idx = self.perm[rank]
        score = self.scores[rank]
        inds = np.flatnonzero(self.words == idx)
        if inds.size == 0:
            return
        
        dists = self.distances[inds]
        kpts = self.kpts[inds]
        imsize = self.img.shape[:2]

        # plot an image grid of the patches
        canvas = self.build_mosaic(kpts, dists, imsize, ncols=None)

        # plot keypoints on the original image
        img = (self.img.copy()*255).astype(np.uint8)
        for i in range(kpts.shape[0]):
            y = kpts[i,0]; x = kpts[i,1]; s = kpts[i,3]*2
            cv2.circle(img, (int(x), int(y)), int(s), (0, 255, 0), 1)

        # Match subplot widths to image aspect ratios so both images have the
        # same displayed height while retaining their native proportions.
        canvas_ratio = canvas.shape[1] / canvas.shape[0]
        img_ratio = img.shape[1] / img.shape[0]
        fig, axes = plt.subplots(1, 2, figsize=figsize,
            gridspec_kw={'width_ratios': [canvas_ratio, img_ratio]},)
        fig.set_constrained_layout(False)
        axes[0].imshow(canvas, aspect='equal')
        axes[0].set_title(f"[Rank {rank+1}] Top patches for visual word {idx} (score: {score:.4f})")
        axes[0].axis('off')
        axes[1].imshow(img, aspect='equal')
        axes[1].set_title("Keypoints on original image")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.perm)

def showEmbeddings2DWithLabels(X, labels, figsize=(14, 6), title=None):

    if isinstance(labels, list):
        labels = np.array(labels)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for category in np.unique(labels):
        mask = labels == category
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=category, alpha=0.7)
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=category, alpha=0.7)

    axes[0].set_title("PCA Embeddings")
    axes[1].set_title("t-SNE Embeddings")

    for ax in axes:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        # ax.legend()
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=16)
    # scale legend based of number of categories and figure size
    n_categories = len(np.unique(labels))
    legend_fontsize = max(8, min(12, int(figsize[0] * 0.5 / n_categories)))
    # place legend outside of the plot
    axes[1].legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()