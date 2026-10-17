from pathlib import Path 
import cv2
import matplotlib
import numpy as np 
matplotlib.use("Agg")    
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402 
from interface import keypoints_to_array  # noqa: E402

GREEN = "#00b400"
RED = "#ff2020"
YELLOW = "#ffd400"
 
def show_image(ax, img, title: str) -> None: 
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.tick_params(labelsize=8)
 
def plot_keypoints(img, kps, title: str, image_label: str): 
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7 * h / w + 0.5), layout="constrained")
    show_image(ax, img, image_label)
    points = keypoints_to_array(kps)
    ax.scatter(points[:, 0], points[:, 1], s=14, facecolors="none", edgecolors=YELLOW, linewidths=0.8)
    fig.suptitle(title, fontsize=12)
    return fig
 
def plot_matches(img1, img2, kps1, kps2, matches, good, label1: str, label2: str, title: str,
                 good_label: str = "correto", bad_label: str = "errado", max_lines: int = 150, seed: int = 0): 
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 13 * h / w + 0.9), layout="constrained")
    show_image(ax1, img1, label1)
    show_image(ax2, img2, label2)
    ax2.yaxis.tick_right()                    
    ax2.yaxis.set_label_position("right")

    good = np.asarray(good, dtype=bool)
    shown = np.arange(len(matches))
    if len(matches) > max_lines:
        shown = np.sort(np.random.default_rng(seed).choice(len(matches), max_lines, replace=False))

    for is_good, color in ((True, GREEN), (False, RED)):   
        for i in shown[good[shown] == is_good]:
            p = kps1[matches[i].queryIdx].pt
            q = kps2[matches[i].trainIdx].pt
            ax1.scatter(*p, s=12, facecolors="none", edgecolors=color, linewidths=0.8)
            ax2.scatter(*q, s=12, facecolors="none", edgecolors=color, linewidths=0.8)
            fig.add_artist(ConnectionPatch(xyA=p, coordsA=ax1.transData, xyB=q, coordsB=ax2.transData,
                                           color=color, linewidth=0.7, alpha=0.85))

    handles = [Line2D([], [], color=GREEN, lw=2, label=good_label)]
    if (~good).any():
        handles.append(Line2D([], [], color=RED, lw=2, label=bad_label))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=12)
    return fig, len(shown)
 
def plot_curves(series, xlabel: str, ylabel: str, title: str, hlines=(), logx: bool = False,
                logy: bool = False, xticks=None, xticklabels=None, ylim=None): 
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for s in series:
        ax.plot(s["x"], s["y"], s.get("style", "-o"), label=s["label"], color=s.get("color"),
                linewidth=1.6, markersize=5)
    for line in hlines:
        ax.axhline(line["y"], color="black", linestyle=(0, (1, 2)), linewidth=1.4, label=line["label"])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels if xticklabels is not None else [f"{v:g}" for v in xticks])
        ax.xaxis.set_minor_locator(NullLocator())    
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return fig
 
def plot_scatter(points, xlabel: str, ylabel: str, title: str, logx: bool = False): 
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for p in points:
        ax.scatter(p["x"], p["y"], s=80, color=p["color"], marker=p["marker"], label=p["label"], zorder=3)
    ax.legend(fontsize=8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    return fig
 
def plot_test_pattern(pairs: np.ndarray, patch_size: int, title: str): 
    half = patch_size / 2
    fig, ax = plt.subplots(figsize=(5, 5.3), layout="constrained")
    for x1, y1, x2, y2 in pairs:
        ax.plot([x1, x2], [y1, y2], color="#1f77b4", linewidth=0.7, alpha=0.7)
    ax.scatter(pairs[:, [0, 2]].ravel(), pairs[:, [1, 3]].ravel(), s=4, color="black", zorder=3)
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(half + 1, -half - 1)    
    ax.set_aspect("equal")
    ax.set_xlabel("x (px, relativo ao centro do patch)")
    ax.set_ylabel("y (px, relativo ao centro do patch)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig
 
def plot_image_grid(images, titles, title: str, ncols: int = 1):
    nrows = -(-len(images) // ncols)
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, (6.5 * h / w + 0.7) * nrows + 0.5),
                             layout="constrained", squeeze=False)
    for ax, img, image_title in zip(axes.ravel(), images, titles):
        show_image(ax, img, image_title)
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    return fig

def save_figure(fig, path) -> Path: 
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jpg":
        fig.savefig(path, dpi=150, pil_kwargs={"quality": 92})   
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
 
def save_captions(captions, path, title: str) -> None: 
    path = Path(path)
    lines = [f"# Legendas — {title}", "", "Geradas automaticamente com os números da corrida.", ""]
    for figure_path, text in captions:
        lines += [f"**{figure_path.relative_to(path.parent).as_posix()}**", "", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
