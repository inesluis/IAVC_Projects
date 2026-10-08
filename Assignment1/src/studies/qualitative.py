"""Análise qualitativa (D19, D21): figuras de keypoints e de matches certos/errados.  [Pessoa A]

  graf_keypoints.png         keypoints de cada método na GRAF img1
  graf_matches_1to2.png      matches certos (verde) / errados (vermelho), GRAF 1->2
  graf_matches_1to4.png      idem, GRAF 1->4 (mudança de ponto de vista forte)
  synthetic_rot45_matches.png, synthetic_scale2_matches.png   exemplos do teste sintético
"""

from pathlib import Path

from datasets import load_graf_pair
from evaluation import match_errors
from interface import Pipeline
from studies.rot_scale_study import make_pairs
from visualization import draw_keypoints, draw_matches, panel_grid, save_figure


def _match_panels(pipelines: list[Pipeline], img1, imgk, gray1, grayk, H, cfg: dict):
    """Para cada pipeline: imagem de matches coloridos + título com as contagens."""
    eps, imgs, titles = cfg["eval"]["eps_px"], [], []
    for p in pipelines:
        f1, fk = p.extract(gray1, cfg), p.extract(grayk, cfg)
        mr = p.match(f1, fk, cfg)
        good = match_errors(f1, fk, mr.matches, H) < eps
        n, c = len(mr.matches), int(good.sum())
        imgs.append(draw_matches(img1, imgk, f1, fk, mr.matches, good,
                                 max_draw=cfg["qualitative"]["max_matches_drawn"], seed=cfg["seed"]))
        titles.append(f"{p.name}: {n} matches, {c} corretos ({100 * c / n:.0f}%)" if n
                      else f"{p.name}: 0 matches")
    return imgs, titles


def run_qualitative(pipelines: list[Pipeline], cfg: dict) -> list[Path]:
    """Gera as figuras qualitativas em results/figures/. Devolve os caminhos gravados."""
    out = Path(cfg["paths"]["figures"])
    qc = cfg["qualitative"]
    saved = []

    # 1) keypoints na GRAF img1
    pr = load_graf_pair(cfg["paths"]["graf"], 2)
    imgs, titles = [], []
    for p in pipelines:
        f = p.extract(pr.gray1, cfg)
        imgs.append(draw_keypoints(pr.bgr1, f, qc["max_keypoints_drawn"]))
        shown = min(f.n, qc["max_keypoints_drawn"] or f.n)
        titles.append(f"{p.name}: {f.n} keypoints (mostrados {shown} mais fortes)")
    saved.append(save_figure(panel_grid(imgs, titles, ncols=3, panel_width_in=5.5,
                                        suptitle="GRAF img1 — keypoints"), out / "graf_keypoints.png"))

    # 2) matches certos/errados GRAF 1->2 e 1->4
    for k in (2, 4):
        pr = load_graf_pair(cfg["paths"]["graf"], k)
        imgs, titles = _match_panels(pipelines, pr.bgr1, pr.bgrk, pr.gray1, pr.grayk, pr.H, cfg)
        fig = panel_grid(imgs, titles, ncols=2,
                         suptitle=f"GRAF 1→{k} — verde: correto (< {cfg['eval']['eps_px']:g} px com H GT), vermelho: errado")
        saved.append(save_figure(fig, out / f"graf_matches_1to{k}.png"))

    # 3) exemplos do teste sintético
    syn = {(p.category, p.k): p for p in make_pairs(cfg)}
    for (cat, lvl), name, label in ((("rot", 45.0), "synthetic_rot45_matches.png", "rotação 45°"),
                                    (("scale", 2.0), "synthetic_scale2_matches.png", "escala 2×")):
        sp = syn[(cat, lvl)]
        imgs, titles = _match_panels(pipelines, sp.gray1, sp.grayk, sp.gray1, sp.grayk, sp.H, cfg)
        saved.append(save_figure(panel_grid(imgs, titles, ncols=2, suptitle=f"Sintético (GRAF img1) — {label}"),
                                 out / name))
    return saved
