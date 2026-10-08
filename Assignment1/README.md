# Assignment #1 — IAVC 2026/27

Keypoint detectors, feature descriptors and applications (Python/OpenCV).

## Instalação

Requer Python 3.10 ou superior. CPU é suficiente; GPU acelera os métodos aprendidos.

```bash
python -m venv venv
venv\Scripts\activate             
python -m pip install -r requirements.txt
```

Verificar o OpenCV:

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'xfeatures2d'))"
# esperado: 4.13.0 True
```

> **Atenção:** nunca instalar `opencv-python` em conjunto com `opencv-contrib-python`.
> Se outro pacote (ex.: LightGlue) o instalar, corrigir com:
> `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python==4.13.0.92`

<!-- TODO (INÊS): instalação de torch + LightGlue e checkpoints usados. -->

## Dados

Os dados não fazem parte do repositório (`data/` está no `.gitignore`).

| Dataset | Origem | Destino |
|---|---|---|
| HPatches (sequences-release, 116 seqs: 57 `i_*` + 59 `v_*`) | https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip | `data/hpatches-sequences-release/<seq>/` |
| GRAF | material da cadeira | `data/graf/` |
| Panoramas | material da cadeira / fotos próprias | `data/Panorama/` |

Estrutura esperada do HPatches (sem pasta repetida depois de extrair o zip):

```
data/hpatches-sequences-release/
├── i_ajuntament/
│   ├── 1.ppm … 6.ppm
│   └── H_1_2 … H_1_6      # H leva img1 -> imgk
├── v_graffiti/
└── …
```

## Como correr

Implementado:

```bash
python src/main.py --part1 --tag classic        # HPatches (i/v)          -> results/tables/classic_*.csv
python src/main.py --rotscale --tag classic     # sintético rotação/escala -> results/tables/classic_rotscale_*.csv
python src/main.py --robustness --tag classic   # tabela i/v/rot/scale     -> results/tables/classic_robustness.csv
python src/main.py --qualitative                # 1 figura por método/objetivo + legendas -> results/figures/qualitative/
python src/main.py --graf --tag classic         # métricas GRAF 1->2 e 1->3 (por par) -> results/tables/classic_graf_pairs.csv
# opções: --pipelines SIFT ORB ...  --seqs i_ajuntament v_graffiti ...  --max-kps 2000|none
```

<!-- TODO: por implementar -->

```bash
python src/main.py --fast      # estudo de parâmetros do FAST
python src/main.py --brief     # estudo do BRIEF próprio
python src/main.py --part2     # stitching              -> results/panoramas/
python src/main.py --all       # tudo
```

Todos os parâmetros estão em `src/config.py`.

## Estrutura

```
src/
├── config.py          parâmetros únicos
├── interface.py       contrato comum (Features, MatchResult, Pipeline)
├── datasets.py        leitura de HPatches / GRAF / panoramas
├── detectors.py       detetores clássicos
├── descriptors.py     descritores clássicos
├── brief_own.py       BRIEF próprio (G I–V)
├── matching.py        ratio test + cross-check, MNN
├── learned.py         SuperPoint, LightGlue
├── homography.py      RANSAC
├── evaluation.py      métricas e corrida no HPatches
├── stitching.py       panoramas
├── methods.py         registo de pipelines
├── visualization.py   figuras
├── studies/           estudos FAST, BRIEF, RANSAC, rotação/escala sintética
└── main.py            ponto de entrada
tests/                 testes de sanidade
results/               tables/, figures/, panoramas/
```
