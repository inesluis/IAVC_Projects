"""Análise qualitativa (D19, D21): uma figura por método e por objetivo, mais legendas prontas.  [Pessoa A]

results/figures/qualitative/
  keypoints/<método>.jpg      keypoints na GRAF imagem 1
  graf_1to2/<método>.jpg      matches certos (verde) / errados (vermelho), GRAF imagem 1 -> imagem 2
  graf_1to4/<método>.jpg      idem, GRAF imagem 1 -> imagem 4 (mudança de ponto de vista forte)
  rot45/<método>.jpg          idem, par sintético rotação 45°
  scale2/<método>.jpg         idem, par sintético escala 2x
  legendas.md                 legenda sugerida para cada figura
"""

from pathlib import Path

from datasets import load_graf_pair
from evaluation import match_errors
from interface import Pipeline
from studies.rot_scale_study import make_pairs
from visualization import plot_keypoints, plot_matches, save_figure


def _slug(name: str) -> str:
    """Nome de ficheiro seguro (ex.: "FAST+BRIEF" -> "FAST-BRIEF")."""
    return name.replace("+", "-")


def _pct(c: int, n: int) -> str:
    return f"{100 * c / n:.0f} %" if n else "—"


def _num(x: float) -> str:
    """Número com vírgula decimal (português)."""
    return f"{x:g}".replace(".", ",")


def _matching_protocol(cfg: dict) -> str:
    m = cfg["matching"]
    return f"ratio test {_num(m['ratio'])}" + (" + cross-check" if m.get("cross_check", True) else "")


def _match_figures(pipelines, group: str, pair_label: str, label1: str, label2: str,
                   img1, img2, gray1, gray2, H, cfg, out_dir):
    """Uma figura de matches por pipeline. Devolve [(caminho, legenda)]."""
    eps = cfg["eval"]["eps_px"]
    res = []
    for p in pipelines:
        f1, f2 = p.extract(gray1, cfg), p.extract(gray2, cfg)
        mr = p.match(f1, f2, cfg)
        good = match_errors(f1, f2, mr.matches, H) < eps
        n, c = len(mr.matches), int(good.sum())
        title = f"{p.name} — {pair_label}: {c} de {n} matches corretos ({_pct(c, n)})"
        fig, shown = plot_matches(img1, img2, f1, f2, mr.matches, good, label1, label2, title,
                                  good_label=f"correto (erro < {_num(eps)} px)", bad_label="errado",
                                  max_draw=cfg["qualitative"]["max_matches_drawn"], seed=cfg["seed"])
        path = save_figure(fig, out_dir / group / f"{_slug(p.name)}.jpg")
        drawn = "todas as linhas mostradas" if shown == n else \
            f"mostradas {shown} de {n} linhas (amostra aleatória que mantém a proporção certos/errados)"
        caption = (f"Matches putativos de {p.name} entre a {label1.lower()} (esquerda) e a {label2.lower()} "
                   f"(direita) — {pair_label} ({_matching_protocol(cfg)}). Verde: correto (erro com a "
                   f"homografia ground-truth < {_num(eps)} px); vermelho: errado. {n} matches, {c} corretos "
                   f"({_pct(c, n)}); {drawn}. Eixos em píxeis.")
        res.append((path, caption))
    return res


def write_captions(entries: list[tuple[Path, str]], path: Path) -> Path:
    """Grava as legendas em Markdown: caminho relativo + texto."""
    lines = ["# Legendas das figuras qualitativas", "",
             "Geradas automaticamente por `python src/main.py --qualitative`. Os números são os desta corrida.", ""]
    for p, cap in entries:
        lines += [f"**{p.relative_to(path.parent).as_posix()}**", "", cap, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_qualitative(pipelines: list[Pipeline], cfg: dict) -> list[Path]:
    """Gera todas as figuras qualitativas e o legendas.md. Devolve os caminhos gravados."""
    out = Path(cfg["paths"]["figures"]) / "qualitative"
    entries: list[tuple[Path, str]] = []
    budget = cfg.get("max_keypoints")

    # 1) keypoints na GRAF imagem 1
    pr = load_graf_pair(cfg["paths"]["graf"], 2)
    h, w = pr.gray1.shape
    for p in pipelines:
        f = p.extract(pr.gray1, cfg)
        fig = plot_keypoints(pr.bgr1, f, f"{p.name} — {f.n} keypoints", "GRAF — imagem 1")
        path = save_figure(fig, out / "keypoints" / f"{_slug(p.name)}.jpg")
        lim = f"limite de {budget} pontos com maior resposta" if budget else "sem limite de pontos"
        n_det = f.meta.get("n_detected", f.n)
        removed = (f"; o detetor {f.meta.get('detector', '')} encontrou {n_det} e o descritor "
                   f"{f.meta.get('descriptor', '')} descartou {n_det - f.n} junto à borda") if n_det != f.n else ""
        entries.append((path, f"Keypoints de {p.name} na imagem 1 do GRAF ({w}×{h} px): {f.n} pontos "
                              f"({lim}{removed}). Cada círculo marca a posição de um keypoint; o tamanho do "
                              f"círculo é fixo e não representa a escala nem a orientação. Eixos em píxeis."))

    # 2) matches GRAF imagem 1 -> imagem k
    for k in (2, 4):
        pr = load_graf_pair(cfg["paths"]["graf"], k)
        entries += _match_figures(pipelines, f"graf_1to{k}", f"GRAF 1→{k}", "Imagem 1", f"Imagem {k}",
                                  pr.bgr1, pr.bgrk, pr.gray1, pr.grayk, pr.H, cfg, out)

    # 3) pares sintéticos (D20): imagem 1 = base, imagem 2 = transformada
    syn = {(p.category, p.k): p for p in make_pairs(cfg)}
    for (cat, lvl), group, label1, label2, pair_label in (
            (("rot", 45.0), "rot45", "Imagem 1 (GRAF imagem 1, máscara circular)",
             "Imagem 2 (rodada 45°)", "rotação sintética de 45°"),
            (("scale", 2.0), "scale2", "Imagem 1 (GRAF imagem 1)",
             "Imagem 2 (escala 2×)", "escala sintética 2×")):
        sp = syn[(cat, lvl)]
        entries += _match_figures(pipelines, group, pair_label, label1, label2,
                                  sp.gray1, sp.grayk, sp.gray1, sp.grayk, sp.H, cfg, out)

    cap = write_captions(entries, out / "legendas.md")
    return [p for p, _ in entries] + [cap]
