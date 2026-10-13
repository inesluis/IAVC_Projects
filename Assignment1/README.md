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

Um comando por experiência (Parte 1). Os resultados ficam em `results/tables/` e `results/figures/`.

```bash
python src/main.py --part1 --tag classic        # HPatches: iluminação e ponto de vista (~15 min)
python src/main.py --rotscale --tag classic     # teste sintético de rotação e escala
python src/main.py --robustness --tag classic   # tabela com as 4 categorias (precisa dos dois anteriores)
python src/main.py --graf --tag classic         # métricas nos pares GRAF 1->2 e 1->3
python src/main.py --qualitative                # figuras de keypoints e de matches, com legendas
python src/main.py --fast --tag classic         # estudo do FAST: threshold x NMS contra o SIFT (~7 min)
python src/main.py --brief --tag classic        # BRIEF próprio (G I, II, III, V) contra o BRIEF do OpenCV
python src/main.py --curves --tag classic       # gráficos a partir das tabelas já calculadas
```

Opções: `--tag NOME` (prefixo dos ficheiros de resultados) e `--seqs SEQ1 SEQ2 ...` (só algumas
sequências do HPatches, para testar depressa).

Testes de sanidade (precisam das imagens GRAF): `python tests/test_sanity.py`

<!-- TODO (INÊS): --part2 (stitching) -->

Todos os parâmetros estão em `src/config.py`. As decisões e justificações estão em `DECISOES.md`,
as verificações contra os artigos e o OpenCV em `VERIFICACOES.md`, e as fontes em `FONTES.md`.

## Estrutura

```
src/
├── config.py          todos os parâmetros
├── interface.py       formato comum do resultado de uma imagem (Features)
├── datasets.py        leitura de HPatches, GRAF e panoramas
├── detectors.py       SIFT, ORB, KAZE, FAST
├── descriptors.py     descritores e extração completa (detetar + descrever)
├── brief_own.py       BRIEF próprio (G I, G II, G III, G V)
├── matching.py        ratio test + cross-check
├── methods.py         lista de métodos; extrair e emparelhar
├── evaluation.py      métricas, corrida sobre os pares, tabelas-resumo
├── visualization.py   figuras
├── studies/           rotação/escala sintética, FAST, BRIEF, figuras qualitativas, gráficos
├── main.py            ponto de entrada
├── learned.py         SuperPoint, LightGlue          (Pessoa B, por fazer)
├── homography.py      RANSAC                         (Pessoa B, por fazer)
└── stitching.py       panoramas                      (Pessoa B, por fazer)
tests/test_sanity.py   testes de sanidade
results/               tables/, figures/, panoramas/
```
