"""Estudo do limiar do RANSAC e da escolha da imagem de referência.

Limiar: com os mesmos matches, varia-se só o limiar e regista-se o nº de inliers,
o erro de reprojeção e o efeito no panorama.
  results/figures/ransac_study/inlier_ratio.png     rácio de inliers em função do limiar, por método
  results/figures/ransac_study/panoramas.jpg        panoramas lado a lado, um por limiar
  results/panoramas/ransac/<conjunto>_thr<limiar>.jpg

Referência: com as mesmas homografias adjacentes, muda-se só a imagem de referência.
  results/figures/reference_study/<conjunto>.jpg    panoramas lado a lado, um por referência
  results/panoramas/reference/<conjunto>_ref<i>.jpg
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import imwrite
from homography import chain_to_reference, project_points
from methods import extract_features
from stitching import (compute_canvas, estimate_adjacent, image_corners, load_panorama_images, match_adjacent,
                       stitch_images)
from studies.curves import comma, method_styles
from studies.stitching_study import matching_text, ransac_text
from visualization import plot_curves, plot_image_grid, save_figure


def stitch_or_none(images, pairs, ref: int):
    """Panorama com a referência dada, ou None se faltar uma homografia ou o canvas for absurdo."""
    if any(pair["H"] is None for pair in pairs):
        return None
    try:
        return stitch_images(images, chain_to_reference([pair["H"] for pair in pairs], ref))
    except ValueError:
        return None

def run_threshold_study(methods, cfg: dict):
    """Varia o limiar do RANSAC num conjunto de imagens fixo. Devolve (tabela, legendas).

    Uma linha por (método, par, limiar). corner_shift_px mede o efeito na geometria do panorama:
    distância média entre os cantos da imagem projetados com a H deste limiar e com a H do limiar
    do config.
    """
    params = cfg["stitching"]
    set_name, thresholds = params["study_set"], params["ransac_thresholds_px"]
    names, images, grays = load_panorama_images(params["sets"][set_name], cfg)
    ref = len(images) // 2
    folder = Path(cfg["paths"]["figures"]) / "ransac_study"

    rows, panoramas = [], {}
    for method in methods:
        features = [extract_features(method, gray, cfg) for gray in grays]
        matches = match_adjacent(method, features, cfg) # os mesmos matches para todos os limiares
        baseline = estimate_adjacent(features, matches, cfg)

        for thr in thresholds:
            cfg_thr = copy.deepcopy(cfg)
            cfg_thr["ransac"]["thr_px"] = thr
            pairs = estimate_adjacent(features, matches, cfg_thr)
            for i, (pair, base) in enumerate(zip(pairs, baseline)):
                shift = np.nan
                if pair["H"] is not None and base["H"] is not None:
                    corners = image_corners(images[i])
                    shift = np.linalg.norm(project_points(corners, pair["H"])
                                           - project_points(corners, base["H"]), axis=1).mean()
                rows.append({"set": set_name, "method": method, "pair": f"{i + 1}->{i + 2}", "thr_px": thr,
                             "N_matches": pair["N_matches"], "N_inliers": pair["N_inliers"],
                             "inlier_ratio": pair["inlier_ratio"], "reproj_err_px": pair["reproj_err_px"],
                             "corner_shift_px": shift})
            if method == params["study_method"]:
                panoramas[thr] = (stitch_or_none(images, pairs, ref), pairs)

    df = pd.DataFrame(rows)
    captions = []

    # rácio de inliers em função do limiar: média dos pares adjacentes, uma linha por método
    means = df.groupby(["method", "thr_px"])["inlier_ratio"].mean()
    styles = method_styles(methods)
    series = [{"x": thresholds, "y": [means.loc[(m, thr)] for thr in thresholds], "label": m,
               "style": "-" + styles[m]["marker"], "color": styles[m]["color"]} for m in methods]
    fig = plot_curves(series, xlabel="Limiar do RANSAC (px, eixo logarítmico)", ylabel="Rácio de inliers",
                      title=f"Rácio de inliers vs limiar do RANSAC — {set_name}", logx=True,
                      xticks=thresholds, xticklabels=[comma(thr) for thr in thresholds], ylim=(-0.03, 1.03))
    path = save_figure(fig, folder / "inlier_ratio.png")
    n_pairs = len(images) - 1
    captions.append((path, f"Rácio de inliers (inliers / matches putativos) em função do limiar do RANSAC, "
                           f"no conjunto {set_name}: média dos {n_pairs} pares de imagens adjacentes. Os "
                           f"matches de cada método são os mesmos em todos os limiares; só muda o limiar. "
                           f"RANSAC com no máximo {cfg['ransac']['max_iters']} iterações e confiança "
                           f"{comma(cfg['ransac']['confidence'])}."))

    # panoramas lado a lado, um por limiar, com o método do estudo
    method = params["study_method"]
    shown, titles = [], []
    for thr, (panorama, pairs) in panoramas.items():
        if panorama is None:
            continue
        imwrite(Path(cfg["paths"]["panoramas"]) / "ransac" / f"{set_name}_thr{thr:g}.jpg", panorama)
        shown.append(panorama)
        titles.append(f"limiar {comma(thr)} px — inliers: " + ", ".join(str(p["N_inliers"]) for p in pairs))
    if shown:
        fig = plot_image_grid(shown, titles, f"{method} — {set_name}: panorama em função do limiar do RANSAC",
                              ncols=2)
        path = save_figure(fig, folder / "panoramas.jpg")
        captions.append((path, f"Panoramas do conjunto {set_name} com {method} "
                               f"({matching_text(method, cfg)}), um por limiar do RANSAC. Referência: imagem "
                               f"central ({names[ref]}); média dos píxeis na sobreposição. No título, o nº de "
                               f"inliers de cada par adjacente. Eixos em píxeis."))
    return df, captions

def polygon_area(points: np.ndarray) -> float:
    """Área de um polígono dado pelos vértices (N, 2), pela fórmula do atacador."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def run_reference_study(cfg: dict):
    """Compara todas as escolhas de imagem de referência em cada conjunto. Devolve (tabela, legendas).

    As homografias adjacentes são as mesmas; só muda a composição. Uma linha por (conjunto, referência):
    tamanho do canvas e ampliação de área de cada imagem (área depois do warp / área original). Uma
    ampliação longe de 1 quer dizer que a imagem ficou esticada ou encolhida, e que o erro das
    homografias compostas é ampliado na mesma proporção.
    """
    params = cfg["stitching"]
    method = params["study_method"]
    rows, captions = [], []

    for set_name, folder in params["sets"].items():
        names, images, grays = load_panorama_images(folder, cfg)
        features = [extract_features(method, gray, cfg) for gray in grays]
        pairs = estimate_adjacent(features, match_adjacent(method, features, cfg), cfg)
        if any(pair["H"] is None for pair in pairs):
            print(f"{set_name}: homografia em falta com {method}; conjunto ignorado")
            continue
        adjacent = [pair["H"] for pair in pairs]

        shown, titles = [], []
        for ref in range(len(images)):
            homographies = chain_to_reference(adjacent, ref)
            scales = [polygon_area(project_points(image_corners(img), H)) / (img.shape[0] * img.shape[1])
                      for img, H in zip(images, homographies)]
            row = {"set": set_name, "method": method, "ref": ref + 1, "ref_image": names[ref],
                   "max_compositions": max(ref, len(images) - 1 - ref)}
            row.update({f"area_scale_{i + 1}": s for i, s in enumerate(scales)})
            try:
                _, (width, height) = compute_canvas(images, homographies)
                panorama = stitch_images(images, homographies)
            except ValueError as error:
                print(f"{set_name}, referência {ref + 1}: {error}")
                rows.append(row)
                continue
            row.update({"canvas_w": width, "canvas_h": height, "canvas_mpx": width * height / 1e6})
            rows.append(row)

            imwrite(Path(cfg["paths"]["panoramas"]) / "reference" / f"{set_name}_ref{ref + 1}.jpg", panorama)
            shown.append(panorama)
            titles.append(f"referência: imagem {ref + 1} ({names[ref]})\ncanvas {width}×{height} px, "
                          f"ampliação de área até {comma(round(max(scales), 2))}×")
        if shown:
            fig = plot_image_grid(shown, titles, f"{method} — {set_name}: escolha da imagem de referência")
            path = save_figure(fig, Path(cfg["paths"]["figures"]) / "reference_study" / f"{set_name}.jpg")
            captions.append((path, f"Panoramas do conjunto {set_name} com {method} "
                                   f"({matching_text(method, cfg)}; {ransac_text(cfg)}), um por imagem de "
                                   f"referência. As homografias entre imagens adjacentes são as mesmas em "
                                   f"todos; as imagens que não são adjacentes à referência chegam lá por "
                                   f"composição de homografias. Média dos píxeis na sobreposição. Eixos em "
                                   f"píxeis."))
    return pd.DataFrame(rows), captions
