"""Registo PIPELINES: nome -> Pipeline (extrator + matcher). Clássicos [A], aprendidos [B].

# PENDENTE (2026-10-08): registar SuperPoint+NN, SuperPoint+LightGlue e SIFT+LightGlue quando
# learned.py (Pessoa B) estiver implementado. Ver DECISOES.md, P06.
"""

from descriptors import CLASSIC_COMBOS, make_extractor
from interface import Pipeline
from matching import match_descriptors

PIPELINES: dict[str, Pipeline] = {
    combo: Pipeline(name=combo, extract=make_extractor(combo), match=match_descriptors)
    for combo in CLASSIC_COMBOS
}


def register(p: Pipeline) -> None:
    """Acrescenta um pipeline ao registo (ex.: métodos aprendidos, BRIEF próprio)."""
    if p.name in PIPELINES:
        raise ValueError(f"Pipeline já registado: {p.name}")
    PIPELINES[p.name] = p


def get_pipelines(names=None) -> list[Pipeline]:
    """Pipelines pela ordem pedida (todos se names=None)."""
    if names is None:
        return list(PIPELINES.values())
    missing = [n for n in names if n not in PIPELINES]
    if missing:
        raise KeyError(f"Pipelines desconhecidos: {missing} (disponíveis: {list(PIPELINES)})")
    return [PIPELINES[n] for n in names]
