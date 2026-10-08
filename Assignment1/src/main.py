"""Ponto de entrada por linha de comando: --part1 | --fast | --brief | --part2 | --all.

Exemplos:
    python src/main.py --part1                                  # HPatches completo, todos os pipelines
    python src/main.py --part1 --seqs i_ajuntament v_graffiti   # teste rápido
    python src/main.py --part1 --pipelines SIFT ORB --max-kps none --tag nolimit

# PENDENTE (2026-10-08): --fast e --brief (Pessoa A, por implementar), --part2 (Pessoa B).
"""

import argparse

from config import get_config, setup_env
from evaluation import run_hpatches, save_tables, summarize
from methods import get_pipelines


def run_part1(args) -> None:
    """Avaliação no HPatches -> results/tables/<tag>raw.csv e tabelas agregadas."""
    max_kps = None if str(args.max_kps).lower() == "none" else int(args.max_kps)
    cfg = get_config(max_keypoints=max_kps)
    setup_env(cfg)
    prefix = f"{args.tag}_" if args.tag else ""
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
    ap.add_argument("--pipelines", nargs="+", default=None, help="subconjunto de pipelines (omissão: todos)")
    ap.add_argument("--seqs", nargs="+", default=None, help="subconjunto de sequências (omissão: todas)")
    ap.add_argument("--max-kps", default=get_config()["max_keypoints"], help="orçamento de keypoints ou 'none'")
    ap.add_argument("--tag", default="", help="prefixo dos ficheiros de resultados")
    args = ap.parse_args()
    if args.part1:
        run_part1(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
