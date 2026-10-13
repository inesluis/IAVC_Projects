"""Ponto de entrada: um comando por experiência. Os resultados ficam em results/.

    python src/main.py --part1          # HPatches: iluminação e ponto de vista (~15 min)
    python src/main.py --rotscale       # teste sintético de rotação e escala
    python src/main.py --robustness     # tabela com as 4 categorias (precisa dos dois anteriores)
    python src/main.py --graf           # métricas nos pares GRAF 1->2 e 1->3
    python src/main.py --qualitative    # figuras de keypoints e de matches
    python src/main.py --fast           # estudo do FAST (~7 min)
    python src/main.py --brief          # BRIEF próprio contra o BRIEF do OpenCV
    python src/main.py --curves         # gráficos a partir das tabelas já calculadas

Opções: --tag NOME (prefixo dos ficheiros) e --seqs SEQ1 SEQ2 ... (só algumas sequências, para testar).

# PENDENTE (2026-10-08): --part2 (stitching, Pessoa B).
"""

import argparse

import pandas as pd

from config import get_config, set_seeds
from datasets import hpatches_pairs, load_graf_pairs
from evaluation import evaluate_pairs, robustness_table, save_tables, summarize
from methods import BRIEF_OWN_METHODS, CLASSIC_METHODS
from studies.brief_study import plot_patterns
from studies.curves import plot_level_curves, plot_tradeoff
from studies.fast_study import plot_fast, run_fast_study, summarize_fast
from studies.qualitative import run_qualitative
from studies.rot_scale_study import make_pairs
from visualization import save_captions


def run_part1(cfg, prefix, seqs):
    """Avalia os métodos clássicos em todos os pares do HPatches."""
    df = evaluate_pairs(CLASSIC_METHODS, hpatches_pairs(cfg, seqs), cfg)
    df.to_csv(cfg["paths"]["tables"] / f"{prefix}raw.csv", index=False)
    tables = summarize(df)
    save_tables(tables, cfg["paths"]["tables"], prefix)
    print(tables["T2_matching"].round(3).to_string())


def run_rotscale(cfg, prefix):
    """Avalia os métodos clássicos no teste sintético de rotação e escala."""
    df = evaluate_pairs(CLASSIC_METHODS, make_pairs(cfg), cfg)
    df.to_csv(cfg["paths"]["tables"] / f"{prefix}rotscale_raw.csv", index=False)
    tables = summarize(df)
    save_tables(tables, cfg["paths"]["tables"], prefix + "rotscale_")
    print(tables["T2_matching"].round(3).to_string())


def run_robustness(cfg, prefix):
    """Junta as duas corridas anteriores numa tabela método x (i, v, rot, scale)."""
    tables_dir = cfg["paths"]["tables"]
    df = pd.concat([pd.read_csv(tables_dir / f"{prefix}raw.csv"),
                    pd.read_csv(tables_dir / f"{prefix}rotscale_raw.csv")], ignore_index=True)
    table = robustness_table(summarize(df))
    table.to_csv(tables_dir / f"{prefix}robustness.csv", float_format="%.4f")
    print(table.round(3).to_string())


def run_graf(cfg, prefix):
    """Métricas nos dois pares GRAF. São só 2 pares: mostra-se cada um, sem médias (D23)."""
    df = evaluate_pairs(CLASSIC_METHODS, load_graf_pairs(cfg), cfg)
    df.to_csv(cfg["paths"]["tables"] / f"{prefix}graf_raw.csv", index=False)
    df["pair"] = "1->" + df["k"].astype(str)
    columns = ["N_features", "N_putative", "N_correct", "N_corresp", "PMR", "precision", "matching_score",
               "recall", "repeatability", "t_det_ms", "t_desc_ms", "t_match_ms"]
    table = df.set_index(["method", "pair"])[columns]
    table.to_csv(cfg["paths"]["tables"] / f"{prefix}graf_pairs.csv", float_format="%.4f")
    print(table.round(3).to_string())


def run_fast(cfg, prefix, seqs):
    """Estudo do FAST: tabela T6, gráficos e legendas."""
    df = run_fast_study(cfg, seqs)
    df.to_csv(cfg["paths"]["tables"] / f"{prefix}fast_raw.csv", index=False)
    table = summarize_fast(df)
    table.to_csv(cfg["paths"]["tables"] / f"{prefix}fast_T6.csv", float_format="%.4f")
    print(table.round(3).to_string())
    folder = cfg["paths"]["figures"] / "fast_study"
    save_captions(plot_fast(df, cfg, folder), folder / "legendas.md", "estudo do FAST")


def run_brief(cfg, prefix, seqs):
    """BRIEF próprio (4 estratégias) contra o BRIEF do OpenCV, com os mesmos keypoints FAST,
    nas 4 categorias: tabela T7 e figuras dos padrões de teste."""
    methods = ["FAST+BRIEF"] + BRIEF_OWN_METHODS
    df = pd.concat([evaluate_pairs(methods, hpatches_pairs(cfg, seqs), cfg),
                    evaluate_pairs(methods, make_pairs(cfg), cfg)], ignore_index=True)
    df.to_csv(cfg["paths"]["tables"] / f"{prefix}brief_raw.csv", index=False)
    tables = summarize(df)
    save_tables({name: tables[name] for name in ("T2_matching", "T5_cost", "F1_by_k")},
                cfg["paths"]["tables"], prefix + "brief_")
    t7 = robustness_table(tables).reindex(methods)
    t7.to_csv(cfg["paths"]["tables"] / f"{prefix}brief_T7.csv", float_format="%.4f")
    print(t7.round(3).to_string())
    folder = cfg["paths"]["figures"] / "brief_own"
    save_captions(plot_patterns(cfg, folder), folder / "legendas.md", "BRIEF próprio")


def run_curves(cfg, prefix):
    """Gráficos F1 e F2 dos métodos clássicos e, se existir, do estudo do BRIEF próprio."""
    tables_dir, figures_dir = cfg["paths"]["tables"], cfg["paths"]["figures"]
    jobs = {
        "curves": [f"{prefix}raw.csv", f"{prefix}rotscale_raw.csv"],
        "curves_brief": [f"{prefix}brief_raw.csv"],
    }
    for name, files in jobs.items():
        if not all((tables_dir / f).exists() for f in files):
            print(f"{name}: faltam as tabelas {files}; correr primeiro as experiências")
            continue
        df = pd.concat([pd.read_csv(tables_dir / f) for f in files], ignore_index=True)
        folder = figures_dir / name
        captions = plot_level_curves(df, cfg, folder) + plot_tradeoff(df, cfg, folder)
        save_captions(captions, folder / "legendas.md", name)
        print(f"{name}: {len(captions)} figuras em {folder}")


def main():
    parser = argparse.ArgumentParser(description="Assignment #1 IAVC")
    for name in ("part1", "rotscale", "robustness", "graf", "qualitative", "fast", "brief", "curves"):
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--tag", default="", help="prefixo dos ficheiros de resultados")
    parser.add_argument("--seqs", nargs="+", default=None, help="só estas sequências do HPatches")
    args = parser.parse_args()

    cfg = get_config()
    set_seeds(cfg)
    prefix = args.tag + "_" if args.tag else ""

    if args.part1:
        run_part1(cfg, prefix, args.seqs)
    if args.rotscale:
        run_rotscale(cfg, prefix)
    if args.robustness:
        run_robustness(cfg, prefix)
    if args.graf:
        run_graf(cfg, prefix)
    if args.qualitative:
        run_qualitative(CLASSIC_METHODS, cfg)
    if args.fast:
        run_fast(cfg, prefix, args.seqs)
    if args.brief:
        run_brief(cfg, prefix, args.seqs)
    if args.curves:
        run_curves(cfg, prefix)


if __name__ == "__main__":
    main()
