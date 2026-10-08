"""Ponto de entrada por linha de comando: --part1 | --fast | --brief | --part2 | --all.

Exemplos:
    python src/main.py --part1                                  # HPatches completo, todos os pipelines
    python src/main.py --part1 --seqs i_ajuntament v_graffiti   # teste rápido
    python src/main.py --part1 --pipelines SIFT ORB --max-kps none --tag nolimit
    python src/main.py --rotscale                               # teste sintético rotação/escala (GRAF img1)
    python src/main.py --robustness --tag classic               # junta i, v, rot, scale numa tabela
    python src/main.py --qualitative                            # figuras GRAF + sintético -> results/figures/

# PENDENTE (2026-10-08): --fast e --brief (Pessoa A, por implementar), --part2 (Pessoa B).
"""

import argparse

import pandas as pd

from config import get_config, setup_env
from evaluation import run_hpatches, save_tables, summarize
from methods import get_pipelines
from studies.qualitative import run_qualitative
from studies.rot_scale_study import run_rot_scale


def _cfg_from_args(args) -> dict:
    max_kps = None if str(args.max_kps).lower() == "none" else int(args.max_kps)
    cfg = get_config(max_keypoints=max_kps)
    setup_env(cfg)
    return cfg


def _prefix(tag: str) -> str:
    return f"{tag}_" if tag else ""


def run_rotscale(args) -> None:
    """Teste sintético -> results/tables/<tag>_rotscale_raw.csv e tabelas (categorias rot/scale)."""
    cfg = _cfg_from_args(args)
    prefix = _prefix(args.tag) + "rotscale_"
    df = run_rot_scale(get_pipelines(args.pipelines), cfg)
    df.to_csv(cfg["paths"]["tables"] / f"{prefix}raw.csv", index=False)
    tables = summarize(df)
    save_tables(tables, cfg["paths"]["tables"], prefix)
    print(tables["T2_matching"].round(3).to_string())


def run_robustness(args) -> None:
    """Junta <tag>_raw.csv (HPatches) e <tag>_rotscale_raw.csv numa tabela pipeline x {i, v, rot, scale}."""
    tables_dir = get_config()["paths"]["tables"]
    p = _prefix(args.tag)
    df = pd.concat([pd.read_csv(tables_dir / f"{p}raw.csv", keep_default_na=False, na_values=[""]),
                    pd.read_csv(tables_dir / f"{p}rotscale_raw.csv", keep_default_na=False, na_values=[""])],
                   ignore_index=True)
    df["error"] = df["error"].fillna("")
    df["control"] = df["control"].fillna(False).astype(bool)
    tables = summarize(df)
    rob = tables["T2_matching"][["PMR", "precision", "matching_score", "recall"]].join(
        tables["T1_detection"][["repeatability"]]).unstack("category")
    rob = rob.reindex(columns=["i", "v", "rot", "scale"], level=1)
    rob.to_csv(tables_dir / f"{p}robustness.csv", float_format="%.4f")
    print(rob.round(3).to_string())


def run_part1(args) -> None:
    """Avaliação no HPatches -> results/tables/<tag>raw.csv e tabelas agregadas."""
    cfg = _cfg_from_args(args)
    prefix = _prefix(args.tag)
    tables_dir = cfg["paths"]["tables"]
    df = run_hpatches(get_pipelines(args.pipelines), cfg, seqs=args.seqs,
                      out_csv=tables_dir / f"{prefix}raw.csv")
    tables = summarize(df)
    save_tables(tables, tables_dir, prefix)
    print(tables["T2_matching"].round(3).to_string())
    if len(tables["errors"]):
        print(f"\nATENÇÃO: {len(tables['errors'])} linhas com erro (ver {prefix}errors.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assignment #1 IAVC — experiências")
    ap.add_argument("--part1", action="store_true", help="avaliação no HPatches")
    ap.add_argument("--rotscale", action="store_true", help="teste sintético de rotação e escala")
    ap.add_argument("--robustness", action="store_true", help="tabela i/v/rot/scale a partir dos CSV")
    ap.add_argument("--qualitative", action="store_true", help="figuras de keypoints e matches (GRAF, sintético)")
    ap.add_argument("--pipelines", nargs="+", default=None, help="subconjunto de pipelines (omissão: todos)")
    ap.add_argument("--seqs", nargs="+", default=None, help="subconjunto de sequências (omissão: todas)")
    ap.add_argument("--max-kps", default=get_config()["max_keypoints"], help="orçamento de keypoints ou 'none'")
    ap.add_argument("--tag", default="", help="prefixo dos ficheiros de resultados")
    args = ap.parse_args()
    if not (args.part1 or args.rotscale or args.robustness or args.qualitative):
        ap.print_help()
        return
    if args.part1:
        run_part1(args)
    if args.rotscale:
        run_rotscale(args)
    if args.robustness:
        run_robustness(args)
    if args.qualitative:
        for path in run_qualitative(get_pipelines(args.pipelines), _cfg_from_args(args)):
            print("figura:", path)


if __name__ == "__main__":
    main()
