"""Força bruta vs vizinho mais próximo aproximado (FLANN) no stitching  

Com os mesmos keypoints e descritores, muda-se só a pesquisa de vizinhos; o ratio test, o cross-check
e o RANSAC são os mesmos. Regista-se o tempo de matching, os matches, os inliers e se a homografia
(e portanto o panorama) muda.
  results/figures/matcher_study/<conjunto>.jpg            panoramas BF e FLANN lado a lado (método do estudo)
  results/panoramas/matcher/<conjunto>_<método>_<bf|flann>.jpg
"""

from pathlib import Path 
import numpy as np
import pandas as pd 
from datasets import imwrite
from homography import corner_distance
from matching import match_descriptors
from methods import extract_features
from stitching import estimate_adjacent, load_panorama_images, stitch_or_none
from studies.curves import comma
from studies.qualitative import file_name
from studies.stitching_study import ransac_text
from visualization import plot_image_grid, save_figure

MATCHERS = {"bf": "força bruta", "flann": "FLANN"}
 
def timed_matches(f1, f2, cfg: dict, matcher: str, repeats: int):
    # Matches de um par e mediana do tempo de matching (ms) em várias medições, após um aquecimento 
    match_descriptors(f1, f2, cfg, matcher)
    runs = [match_descriptors(f1, f2, cfg, matcher) for _ in range(repeats)]
    return runs[0][0], float(np.median([t for _, t in runs]))
 
def run_matcher_study(cfg: dict):
    """Corre BF e FLANN nos métodos de cfg["stitching"]["matcher_study"] (só métodos com descritores:
    o LightGlue não faz pesquisa de vizinhos). Devolve (tabela, legendas).

    Uma linha por (conjunto, método, matcher, par adjacente). Colunas que comparam com a força bruta:
      shared_with_bf   fração dos matches que também existem na força bruta (mesmo queryIdx e trainIdx)
      corner_diff_px   distância média entre os 4 cantos da imagem projetados com a H deste matcher e
                       com a H da força bruta (0 = mesma geometria, logo mesmo panorama)
    """
    params = cfg["stitching"]
    study = params["matcher_study"]
    rows, captions = [], []

    for set_name, folder in params["sets"].items():
        names, images, grays = load_panorama_images(folder, cfg)
        ref = len(images) // 2
        shown, titles = [], []

        for method in study["methods"]:
            features = [extract_features(method, gray, cfg) for gray in grays]
            results = {}
            for matcher in MATCHERS:
                timed = [timed_matches(features[i], features[i + 1], cfg, matcher, study["repeats"])
                         for i in range(len(features) - 1)]
                matches = [m for m, _ in timed]
                results[matcher] = (matches, [t for _, t in timed], estimate_adjacent(features, matches, cfg))

            bf_matches, _, bf_pairs = results["bf"]
            for matcher, (matches, times, pairs) in results.items():
                panorama = stitch_or_none(images, pairs, ref)
                for i, pair in enumerate(pairs):
                    bf_set = {(m.queryIdx, m.trainIdx) for m in bf_matches[i]}
                    shared = (np.mean([(m.queryIdx, m.trainIdx) in bf_set for m in matches[i]])
                              if matches[i] else np.nan)
                    diff = corner_distance(pair["H"], bf_pairs[i]["H"], images[i])
                    rows.append({"set": set_name, "method": method, "matcher": matcher, "pair": f"{i + 1}->{i + 2}",
                                 "t_match_ms": times[i], "N_matches": pair["N_matches"],
                                 "N_inliers": pair["N_inliers"], "inlier_ratio": pair["inlier_ratio"],
                                 "reproj_err_px": pair["reproj_err_px"], "shared_with_bf": shared,
                                 "corner_diff_px": diff,
                                 "panorama_w": panorama.shape[1] if panorama is not None else None,
                                 "panorama_h": panorama.shape[0] if panorama is not None else None})
                if panorama is not None:
                    imwrite(Path(cfg["paths"]["panoramas"]) / "matcher"
                            / f"{set_name}_{Path(file_name(method)).stem}_{matcher}.jpg", panorama)
                    if method == params["study_method"]:
                        shown.append(panorama)
                        titles.append(f"{MATCHERS[matcher]} — tempo de matching: "
                                      + " + ".join(f"{t:.0f}" for t in times) + " ms; inliers: "
                                      + ", ".join(str(p["N_inliers"]) for p in pairs))

        if shown:
            method = params["study_method"]
            fig = plot_image_grid(shown, titles, f"{method} — {set_name}: força bruta vs FLANN")
            path = save_figure(fig, Path(cfg["paths"]["figures"]) / "matcher_study" / f"{set_name}.jpg")
            flann = cfg["matching"]["flann"]
            captions.append((path, f"Panoramas do conjunto {set_name} com {method}, com o matching por força "
                                   f"bruta (cima) e por vizinho aproximado FLANN (baixo; {flann['kdtree_trees']} "
                                   f"KD-trees, {flann['checks']} verificações). Os keypoints e os descritores são "
                                   f"os mesmos; ratio test {comma(cfg['matching']['ratio'])} + cross-check nos "
                                   f"dois; {ransac_text(cfg)}. Referência: imagem central ({names[ref]}); média "
                                   f"dos píxeis na sobreposição. No título, o tempo de matching de cada par "
                                   f"adjacente (mediana de {study['repeats']} medições) e o nº de inliers. "
                                   f"Eixos em píxeis."))
    return pd.DataFrame(rows), captions