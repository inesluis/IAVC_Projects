"""Stitching com várias configurações: tabela T8, figuras e panoramas

Para cada conjunto de imagens e cada método:
  results/figures/stitching/<conjunto>/keypoints/<método>.jpg      keypoints na imagem de referência
  results/figures/stitching/<conjunto>/inliers_<i>-<j>/<método>.jpg  inliers (verde) e outliers do RANSAC
  results/panoramas/<conjunto>/<método>.jpg                        panorama, referência central, média
"""

from pathlib import Path

import pandas as pd

from datasets import imwrite
from homography import chain_to_reference
from methods import extract_features
from stitching import estimate_adjacent, load_panorama_images, match_adjacent, stitch_images
from studies.curves import comma
from studies.qualitative import file_name
from visualization import plot_keypoints, plot_matches, save_figure


def matching_text(method: str, cfg: dict) -> str:
    """Como foram obtidos os matches putativos de um método, para as legendas."""
    if method.endswith("LightGlue"):
        threshold = comma(cfg["learned"]["lightglue"]["filter_threshold"])
        return f"matches do LightGlue (confiança mínima {threshold})"
    return f"ratio test {comma(cfg['matching']['ratio'])} + cross-check"


def ransac_text(cfg: dict) -> str:
    ransac = cfg["ransac"]
    return (f"RANSAC com limiar de {comma(ransac['thr_px'])} px, no máximo {ransac['max_iters']} iterações "
            f"e confiança {comma(ransac['confidence'])}")


def run_stitching(methods, cfg: dict):
    """Corre o pipeline completo em todos os conjuntos. Devolve (tabela, legendas).

    A tabela tem uma linha por (conjunto, método, par de imagens adjacentes) com o nº de matches,
    o nº e o rácio de inliers e o erro médio de reprojeção dos inliers
    """
    figures = Path(cfg["paths"]["figures"]) / "stitching"
    max_lines = cfg["qualitative"]["max_matches_drawn"]
    rows, captions = [], []

    for set_name, folder in cfg["stitching"]["sets"].items():
        names, images, grays = load_panorama_images(folder, cfg)
        ref = len(images) // 2   # imagem central
        h, w = grays[ref].shape

        for method in methods:
            features = [extract_features(method, gray, cfg) for gray in grays]
            matches = match_adjacent(method, features, cfg)
            pairs = estimate_adjacent(features, matches, cfg)

            # panorama: só se todas as homografias foram estimadas e o canvas tem um tamanho razoável
            panorama = None
            if all(pair["H"] is not None for pair in pairs):
                try:
                    panorama = stitch_images(images, chain_to_reference([pair["H"] for pair in pairs], ref))
                    imwrite(Path(cfg["paths"]["panoramas"]) / set_name / file_name(method), panorama)
                except ValueError as error:
                    print(f"{set_name} {method}: {error}")

            fig = plot_keypoints(images[ref], features[ref].kps,
                                 f"{method} — {len(features[ref].kps)} keypoints", f"{set_name} — {names[ref]}")
            path = save_figure(fig, figures / set_name / "keypoints" / file_name(method))
            captions.append((path, f"Keypoints de {method} na imagem de referência do conjunto {set_name} "
                                   f"({names[ref]}, {w}×{h} px): {len(features[ref].kps)} pontos (limite de "
                                   f"{cfg['max_keypoints']}). O tamanho do círculo é fixo. Eixos em píxeis."))

            for i, pair in enumerate(pairs):
                rows.append({
                    "set": set_name, "method": method, "pair": f"{i + 1}->{i + 2}",
                    "image1": names[i], "image2": names[i + 1],
                    "N_kps1": len(features[i].kps), "N_kps2": len(features[i + 1].kps),
                    "N_matches": pair["N_matches"], "N_inliers": pair["N_inliers"],
                    "inlier_ratio": pair["inlier_ratio"], "reproj_err_px": pair["reproj_err_px"],
                    "panorama_w": panorama.shape[1] if panorama is not None else None,
                    "panorama_h": panorama.shape[0] if panorama is not None else None,
                })
                if pair["H"] is None:
                    continue
                n, n_in = pair["N_matches"], pair["N_inliers"]
                title = f"{method} — {set_name} {i + 1}→{i + 2}: {n_in} inliers em {n} matches ({100 * n_in / n:.0f} %)"
                fig, n_lines = plot_matches(images[i], images[i + 1], features[i].kps, features[i + 1].kps,
                                            matches[i], pair["inliers"], names[i], names[i + 1], title,
                                            good_label="inlier do RANSAC", bad_label="outlier",
                                            max_lines=max_lines, seed=cfg["seed"])
                path = save_figure(fig, figures / set_name / f"inliers_{i + 1}-{i + 2}" / file_name(method))
                lines = "todas as linhas mostradas" if n_lines == n else f"mostradas {n_lines} de {n} linhas"
                captions.append((path, f"Correspondências de {method} entre {names[i]} (esquerda) e "
                                       f"{names[i + 1]} (direita), conjunto {set_name}: "
                                       f"{matching_text(method, cfg)}; {ransac_text(cfg)}. Verde: inlier; "
                                       f"vermelho: outlier. {n} matches, {n_in} inliers "
                                       f"({100 * n_in / n:.0f} %), erro médio de reprojeção dos inliers "
                                       f"{comma(round(pair['reproj_err_px'], 2))} px; {lines}. Eixos em píxeis."))
    return pd.DataFrame(rows), captions
