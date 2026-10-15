# Figuras qualitativas: uma figura por método e por objetivo 

from pathlib import Path
from datasets import load_graf_pairs
from evaluation import match_errors
from methods import extract_features, match_features
from studies.rot_scale_study import make_pairs
from visualization import plot_keypoints, plot_matches, save_captions, save_figure

def file_name(method: str) -> str: 
    return method.replace("+", "-") + ".jpg"

def keypoint_figures(methods, pair, cfg: dict, folder: Path) -> list:
    # Uma figura por método com os keypoints da imagem 1 do par 
    h, w = pair.gray1.shape
    limit = f"limite de {cfg['max_keypoints']} pontos com maior resposta"
    captions = []
    for method in methods:
        f = extract_features(method, pair.gray1, cfg)
        fig = plot_keypoints(pair.bgr1, f.kps, f"{method} — {len(f.kps)} keypoints", "GRAF — imagem 1")
        path = save_figure(fig, folder / "keypoints" / file_name(method))

        removed = ""
        if f.n_detected != len(f.kps):
            detector, descriptor = method.split("+")
            removed = (f"; o detetor {detector} encontrou {f.n_detected} e o descritor {descriptor} "
                       f"descartou {f.n_detected - len(f.kps)} junto à borda")
        captions.append((path, f"Keypoints de {method} na imagem 1 do GRAF ({w}×{h} px): {len(f.kps)} pontos "
                               f"({limit}{removed}). Cada círculo marca a posição de um keypoint; o tamanho "
                               f"do círculo é fixo e não representa a escala nem a orientação. Eixos em píxeis."))
    return captions


def match_figures(methods, pair, img1, imgk, label1: str, labelk: str, pair_name: str, cfg: dict,
                  folder: Path) -> list:
    # Uma figura por método com os matches do par, verdes se corretos 
    eps = cfg["eval"]["eps_px"]
    max_lines = cfg["qualitative"]["max_matches_drawn"]
    captions = []
    for method in methods:
        f1 = extract_features(method, pair.gray1, cfg)
        fk = extract_features(method, pair.grayk, cfg)
        matches, _ = match_features(method, f1, fk, cfg)
        good = match_errors(f1, fk, matches, pair.H) < eps

        n, n_good = len(matches), int(good.sum())
        percent = f"{100 * n_good / n:.0f} %" if n else "—"
        title = f"{method} — {pair_name}: {n_good} de {n} matches corretos ({percent})"
        fig, n_lines = plot_matches(img1, imgk, f1.kps, fk.kps, matches, good, label1, labelk, title,
                                    good_label=f"correto (erro < {eps:g} px)", max_lines=max_lines,
                                    seed=cfg["seed"])
        path = save_figure(fig, folder / file_name(method))

        if n_lines == n:
            lines = "todas as linhas mostradas"
        else:
            lines = f"mostradas {n_lines} de {n} linhas (amostra aleatória que mantém a proporção certos/errados)"
        ratio = f"{cfg['matching']['ratio']:g}".replace(".", ",")
        captions.append((path, f"Matches putativos de {method} entre a {label1.lower()} (esquerda) e a "
                               f"{labelk.lower()} (direita) — {pair_name} (ratio test {ratio} + cross-check). "
                               f"Verde: correto (erro com a homografia ground-truth < {eps:g} px); vermelho: "
                               f"errado. {n} matches, {n_good} corretos ({percent}); {lines}. Eixos em píxeis."))
    return captions


def run_qualitative(methods, cfg: dict) -> None:
    # Gera todas as figuras qualitativas e o ficheiro de legendas 
    folder = Path(cfg["paths"]["figures"]) / "qualitative"
    graf = load_graf_pairs(cfg)

    captions = keypoint_figures(methods, graf[0], cfg, folder)

    for pair in graf:   # GRAF imagem 1 -> imagem 2 e imagem 1 -> imagem 3
        captions += match_figures(methods, pair, pair.bgr1, pair.bgrk, "Imagem 1", f"Imagem {pair.k}",
                                  f"GRAF 1→{pair.k}", cfg, folder / f"graf_1to{pair.k}")

    synthetic = {(p.category, p.k): p for p in make_pairs(cfg)}
    rot45, scale2 = synthetic[("rot", 45.0)], synthetic[("scale", 2.0)]
    captions += match_figures(methods, rot45, rot45.gray1, rot45.grayk,
                              "Imagem 1 (GRAF imagem 1, máscara circular)", "Imagem 2 (rodada 45°)",
                              "rotação sintética de 45°", cfg, folder / "rot45")
    captions += match_figures(methods, scale2, scale2.gray1, scale2.grayk,
                              "Imagem 1 (GRAF imagem 1)", "Imagem 2 (escala 2×)",
                              "escala sintética 2×", cfg, folder / "scale2")

    save_captions(captions, folder / "legendas.md", "figuras qualitativas")
    print(f"{len(captions)} figuras em {folder}")
