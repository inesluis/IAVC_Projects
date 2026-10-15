"""Ponto de entrada: um comando por experiência. Os resultados ficam em results/.

    python src/main.py --part1          # HPatches: iluminação e ponto de vista (~15 min)
    python src/main.py --rotscale       # teste sintético de rotação e escala
    python src/main.py --robustness     # tabela com as 4 categorias (precisa dos dois anteriores)
    python src/main.py --graf           # métricas nos pares GRAF 1->2 e 1->3
    python src/main.py --qualitative    # figuras de keypoints e de matches
    python src/main.py --fast           # estudo do FAST (~7 min)
    python src/main.py --brief          # BRIEF próprio contra o BRIEF do OpenCV
    python src/main.py --curves         # gráficos a partir das tabelas já calculadas
    python src/main.py --part2          # stitching: configurações, limiar do RANSAC e imagem de referência

Opções: --tag NOME (prefixo dos ficheiros), --seqs SEQ1 SEQ2 ... (só algumas sequências, para testar)
e --methods M1 M2 ... (só alguns métodos; por omissão correm os clássicos e os aprendidos, e no --part2
os de cfg["stitching"]["methods"]).
"""

import argparse

import pandas as pd

from config import get_config, set_seeds
from datasets import hpatches_pairs, load_graf_pairs
from evaluation import evaluate_pairs, robustness_table, save_tables, summarize
from methods import ALL_METHODS, BRIEF_OWN_METHODS
from studies.brief_study import plot_patterns
from studies.curves import plot_level_curves, plot_tradeoff
from studies.fast_study import plot_fast, run_fast_study, summarize_fast
from studies.qualitative import run_qualitative
from studies.matcher_study import run_matcher_study
from studies.ransac_study import run_reference_study, run_threshold_study
from studies.rot_scale_study import make_pairs
from studies.stitching_study import run_stitching
from visualization import save_captions


def run_part1(cfg, prefix, seqs, methods):
    """Avalia os métodos em todos os pares do HPatches."""
    df = evaluate_pairs(methods, hpatches_pairs(cfg, seqs), cfg)
    df.to_csv(cfg["paths"]["tables"] / f"{prefix}raw.csv", index=False)
    tables = summarize(df)
    save_tables(tables, cfg["paths"]["tables"], prefix)
    print(tables["T2_matching"].round(3).to_string())


def run_rotscale(cfg, prefix, methods):
    """Avalia os métodos no teste sintético de rotação e escala."""
    df = evaluate_pairs(methods, make_pairs(cfg), cfg)
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


def run_graf(cfg, prefix, methods):
    """Métricas nos dois pares GRAF. São só 2 pares: mostra-se cada um, sem médias (D23)."""
    df = evaluate_pairs(methods, load_graf_pairs(cfg), cfg)
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
    """Gráficos F1 e F2 dos métodos avaliados e, se existir, do estudo do BRIEF próprio."""
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


def run_part2(cfg, prefix, methods):
    """Parte 2: stitching com várias configurações (tabela T8), estudo do limiar do RANSAC,
    comparação de imagens de referência e força bruta vs FLANN. Os panoramas ficam em results/panoramas/."""
    tables_dir, figures_dir = cfg["paths"]["tables"], cfg["paths"]["figures"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    table, captions = run_stitching(methods, cfg)
    table.to_csv(tables_dir / f"{prefix}stitching_T8.csv", index=False, float_format="%.4f")
    save_captions(captions, figures_dir / "stitching" / "legendas.md", "stitching")
    print(table.drop(columns=["image1", "image2"]).round(3).to_string(index=False))

    table, captions = run_threshold_study(methods, cfg)
    table.to_csv(tables_dir / f"{prefix}ransac_threshold.csv", index=False, float_format="%.4f")
    save_captions(captions, figures_dir / "ransac_study" / "legendas.md", "limiar do RANSAC")
    print(table[table["method"] == cfg["stitching"]["study_method"]].round(3).to_string(index=False))

    table, captions = run_reference_study(cfg)
    table.to_csv(tables_dir / f"{prefix}reference.csv", index=False, float_format="%.4f")
    save_captions(captions, figures_dir / "reference_study" / "legendas.md", "imagem de referência")
    print(table.round(3).to_string(index=False))

    table, captions = run_matcher_study(cfg)
    table.to_csv(tables_dir / f"{prefix}matcher_bf_flann.csv", index=False, float_format="%.4f")
    save_captions(captions, figures_dir / "matcher_study" / "legendas.md", "força bruta vs FLANN")
    print(table.drop(columns=["panorama_w", "panorama_h"]).round(3).to_string(index=False))


def main():
    parser =argparse.ArgumentParser(description="Assignment #1 IAVC")
    for name in ("part1", "rotscale", "robustness", "graf", "qualitative", "fast", "brief", "curves", "part2"):
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--tag", default="", help="prefixo dos ficheiros de resultados")
    parser.add_argument("--seqs", nargs="+", default=None, help="só estas sequências do HPatches")
    parser.add_argument("--methods", nargs="+", default=None, choices=ALL_METHODS, metavar="MÉTODO",
                        help="só estes métodos (por omissão: todos, clássicos e aprendidos)")
    args = parser.parse_args()

    cfg = get_config()
    set_seeds(cfg)
    prefix = args.tag + "_" if args.tag else ""
    methods = args.methods or ALL_METHODS

    if args.part1:
        run_part1(cfg, prefix, args.seqs, methods)
    if args.rotscale:
        run_rotscale(cfg, prefix, methods)
    if args.robustness:
        run_robustness(cfg, prefix)
    if args.graf:
        run_graf(cfg, prefix, methods)
    if args.qualitative:
        run_qualitative(methods, cfg)
    if args.fast:
        run_fast(cfg, prefix, args.seqs)
    if args.brief:
        run_brief(cfg, prefix, args.seqs)
    if args.curves:
        run_curves(cfg, prefix)
    if args.part2:
        run_part2(cfg, prefix, args.methods or cfg["stitching"]["methods"])


if __name__ == "__main__":
    main()
