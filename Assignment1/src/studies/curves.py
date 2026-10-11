"""Gráficos a partir dos CSV já calculados (não repete corridas).  [Pessoa A]

F1  curvas por nível: uma figura por categoria (i, v, rot, scale) e por métrica, uma linha por método.
    Nível = par 1->k (HPatches), ângulo (rotação) ou fator (escala).
F2  compromisso exatidão vs tempo: tempo total por par (x) contra matching score / precisão (y),
    um ponto por método, para i e v.
Ver DECISOES.md, D26.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualization import plot_curves, plot_scatter, save_figure

METRICS = {"precision": "Precisão", "matching_score": "Matching score",
           "recall": "Recall", "repeatability": "Repetibilidade"}
CATEGORIES = {
    "i": {"name": "iluminação (HPatches i_*)", "xlabel": "Par de imagens"},
    "v": {"name": "ponto de vista (HPatches v_*)", "xlabel": "Par de imagens"},
    "rot": {"name": "rotação sintética (GRAF imagem 1)", "xlabel": "Ângulo de rotação (°)"},
    "scale": {"name": "escala sintética (GRAF imagem 1)", "xlabel": "Fator de escala (eixo logarítmico)"},
}
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def load_raw(paths: list[Path]) -> pd.DataFrame:
    """Junta os CSV por par (HPatches e/ou sintético) e remove as linhas com erro."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df[df["error"].fillna("") == ""]


def method_styles(pipelines: list[str]) -> dict[str, dict]:
    """Cor e marcador fixos por método, iguais em todas as figuras."""
    colors = plt.get_cmap("tab10").colors
    return {p: {"color": colors[i % 10], "marker": _MARKERS[i % len(_MARKERS)]}
            for i, p in enumerate(pipelines)}


def _num(v: float) -> str:
    return f"{v:g}".replace(".", ",")


def _protocol(cfg: dict) -> str:
    kps = cfg.get("max_keypoints")
    lim = f"limite de {kps} keypoints por imagem" if kps else "sem limite de keypoints"
    return (f"Protocolo: {lim}, ratio test {_num(cfg['matching']['ratio'])} + cross-check, match correto se o "
            f"erro com a homografia ground-truth for < {_num(cfg['eval']['eps_px'])} px.")


def plot_level_curves(df: pd.DataFrame, cfg: dict, out_dir: Path) -> list[tuple[Path, str]]:
    """F1: métrica em função do nível, por categoria. Devolve [(caminho, legenda)]."""
    pipelines = list(dict.fromkeys(df["pipeline"]))
    styles = method_styles(pipelines)
    out = []
    for cat, info in CATEGORIES.items():
        d = df[df["category"] == cat]
        if d.empty:
            continue
        levels = sorted(d["k"].unique())
        means = d.groupby(["pipeline", "k"])[list(METRICS)].mean()
        n_per_level = int(d.groupby(["pipeline", "k"]).size().iloc[0])
        if cat in ("i", "v"):
            ticks, labels = levels, [f"1→{int(k)}" for k in levels]
            order = ("a mudança de ponto de vista aumenta de 1→2 para 1→6" if cat == "v" else
                     "as imagens 2 a 6 são condições de iluminação diferentes, sem ordem de dificuldade garantida")
            how = f"Cada ponto é a média de {n_per_level} pares (um por sequência); {order}."
        elif cat == "rot":
            ticks = [k for k in levels if k % 30 == 0]
            labels = [f"{int(k)}" for k in ticks]
            how = "Um par por nível: a imagem base rodada em torno do centro; 0° é o controlo."
        else:
            ticks, labels = levels, [_num(k) for k in levels]
            how = "Um par por nível: a imagem base escalada em torno do centro; 1 é o controlo."
        for col, mname in METRICS.items():
            series = [{"x": levels, "y": [means.loc[(p, k), col] for k in levels], "label": p,
                       "style": "-" + styles[p]["marker"], "color": styles[p]["color"]} for p in pipelines]
            # precisão e repetibilidade: eixo fixo 0-1; matching score e recall: ajustado aos dados
            ylim = (-0.03, 1.03) if col in ("precision", "repeatability") else (-0.01, None)
            fig = plot_curves(series, xlabel=info["xlabel"], ylabel=mname, title=f"{mname} — {info['name']}",
                              logx=(cat == "scale"), xticks=ticks, xticklabels=labels, ylim=ylim)
            path = save_figure(fig, out_dir / f"F1_{col}_{cat}.png")
            out.append((path, f"{mname} de cada método em {info['name']}, em função de: "
                              f"{info['xlabel'].split(' (')[0].lower()}. {how} {_protocol(cfg)}"))
    return out


def plot_tradeoff(df: pd.DataFrame, cfg: dict, out_dir: Path) -> list[tuple[Path, str]]:
    """F2: tempo total por par contra exatidão, um ponto por método, para i e v."""
    pipelines = list(dict.fromkeys(df["pipeline"]))
    styles = method_styles(pipelines)
    out = []
    for cat in ("i", "v"):
        d = df[df["category"] == cat]
        if d.empty:
            continue
        means = d.groupby("pipeline")[["t_total_ms", "matching_score", "precision"]].mean()
        n = int(d.groupby("pipeline").size().iloc[0])
        for col in ("matching_score", "precision"):
            pts = [{"x": means.loc[p, "t_total_ms"], "y": means.loc[p, col], "label": p, **styles[p]}
                   for p in pipelines]
            fig = plot_scatter(pts, xlabel="Tempo total por par (ms, eixo logarítmico)", ylabel=METRICS[col],
                               title=f"{METRICS[col]} vs tempo — {CATEGORIES[cat]['name']}", logx=True)
            path = save_figure(fig, out_dir / f"F2_{col}_vs_time_{cat}.png")
            out.append((path, f"Compromisso entre exatidão e custo em {CATEGORIES[cat]['name']}: "
                              f"{METRICS[col].lower()} médio contra o tempo total médio por par, um ponto por "
                              f"método (média de {n} pares). Tempo total = 2 × (deteção + descrição) + matching, "
                              f"em ms, eixo logarítmico. {_protocol(cfg)}"))
    return out


def write_captions(entries, path: Path, title: str) -> Path:
    lines = [f"# Legendas — {title}", "",
             "Geradas automaticamente por `python src/main.py --curves`, a partir dos CSV da corrida.", ""]
    for p, cap in entries:
        lines += [f"**{p.name}**", "", cap, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
