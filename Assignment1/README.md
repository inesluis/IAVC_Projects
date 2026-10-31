# Assignment #1 — IAVC 2026/27

Detetores de keypoints, descritores locais e aplicações (Python/OpenCV).
Carolina Gonçalves (2022221124) e Inês Luís (2022217106).

## 1. Requisitos

- **Python 3.10 ou superior** (testado com Python 3.14.7).
- **CPU ou GPU.** Os métodos clássicos correm no CPU. Os métodos aprendidos (SuperPoint, SuperGlue,
  LightGlue) correm na GPU se existir e no CPU caso contrário; no CPU funcionam, mas são bastante mais
  lentos. Os resultados entregues foram obtidos com um Intel Core i7-12650H e uma GPU NVIDIA
  GeForce RTX 3050 Ti Laptop (torch 2.14.1 com CUDA 12.6).

## 2. Instalação

```bash
python -m venv venv
venv\Scripts\activate                     # Linux/macOS: source venv/bin/activate
python -m pip install -r requirements.txt
```

**Correção obrigatória do OpenCV.** O LightGlue instala o `opencv-python`, que se sobrepõe ao
`opencv-contrib-python` e remove o KAZE, o BRIEF e o FREAK. Depois do passo anterior, correr:

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'xfeatures2d'))"
# esperado: 4.13.0 True
```

> **Atenção:** nunca instalar `opencv-python` em conjunto com `opencv-contrib-python`.
> Se outro pacote (ex.: LightGlue) o instalar, corrigir com:
> `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python==4.13.0.92`

### Métodos aprendidos (PyTorch + LightGlue)

O `requirements.txt` instala o PyTorch e o LightGlue (repositório [cvg/LightGlue](https://github.com/cvg/LightGlue),
que inclui o SuperPoint). Dois cuidados:

1. O LightGlue depende de `opencv-python`, que o pip instala por cima do contrib. Depois de instalar os
   requisitos, correr a correção indicada acima e repetir a verificação do OpenCV.
2. No Windows, `pip install torch` instala a versão só para CPU. Para usar uma GPU NVIDIA:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   (testado com torch 2.11.0+cu128).

```bash
python -c "import torch, lightglue; print(torch.__version__, torch.cuda.is_available())"
```

A GPU é usada se existir; para forçar o CPU, pôr `CFG["learned"]["device"] = "cpu"` em `src/config.py`.
Os tempos das tabelas dependem desta escolha, por isso indicar sempre onde foram medidos.

Pesos pré-treinados (não se treina nada). São descarregados pelo pacote na primeira utilização, para a
cache do PyTorch (`~/.cache/torch/hub/checkpoints/`), e não fazem parte da entrega:

| Modelo | Ficheiro | Origem |
|---|---|---|
| SuperPoint (detetor + descritor) | `superpoint_v1.pth` | pesos da Magic Leap, release `v0.1_arxiv` do cvg/LightGlue |
| LightGlue para SuperPoint (matcher) | `superpoint_lightglue.pth` | release `v0.1_arxiv` do cvg/LightGlue |
| LightGlue para SIFT (matcher) | `sift_lightglue.pth` | release `v0.1_arxiv` do cvg/LightGlue |

O "SIFT+LightGlue" usa o SIFT do pacote LightGlue (RootSIFT, com os parâmetros com que o matcher foi
treinado), que não é o `cv2.SIFT` da configuração clássica "SIFT".

## Dados

Os dados não fazem parte do repositório (`data/` está no `.gitignore`).

| Dataset | Origem | Destino |
|---|---|---|
| HPatches (sequences-release, 116 seqs: 57 `i_*` + 59 `v_*`) | https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip | `data/hpatches-sequences-release/<seq>/` |
| GRAF | material da cadeira | `data/graf/` |
| Panoramas `keble` (3 imagens) | material da cadeira | `data/Panorama/` |
| Panoramas `proprio` (3 fotografias nossas) | fotos próprias | `data/panorama2/` |

Depois de extrair o zip do HPatches, confirmar que não fica uma pasta repetida:

```
data/hpatches-sequences-release/
├── i_ajuntament/
│   ├── 1.ppm … 6.ppm
│   └── H_1_2 … H_1_6      # homografia verdadeira: imagem 1 -> imagem k
├── v_graffiti/
└── …
```

## Como correr

Um comando por experiência. Os resultados ficam em `results/tables/`, `results/figures/` e `results/panoramas/`.

### Parte 1

```bash
cd src
python main.py --part1 --rotscale --robustness --graf --qualitative --fast --brief --curves --part2 --tag final
```

Ou cada experiência em separado (a ordem importa: `--robustness` precisa de `--part1` e `--rotscale`;
`--curves` precisa de `--part1`, `--rotscale` e `--brief`):

| Comando | Experiência | Secção do relatório |
|---|---|---|
| `--part1` | HPatches: 10 métodos em iluminação e ponto de vista; métricas da homografia | 4.2, 4.6, 4.7 |
| `--rotscale` | Teste sintético de rotação e escala (imagem 1 do GRAF) | 4.3, 4.6 |
| `--robustness` | Tabela com as 4 categorias (i, v, rotação, escala) | 4.3 |
| `--graf` | Métricas nos pares GRAF 1→2 e 1→3 | 4.4 |
| `--qualitative` | Figuras de keypoints e de matches certos e errados | 4.4 |
| `--fast` | Estudo do FAST: threshold × NMS contra o SIFT | 4.1 |
| `--brief` | BRIEF implementado (G I, II, III, V) contra o BRIEF do OpenCV | 4.5 |
| `--curves` | Gráficos a partir das tabelas já calculadas | 4.3, 4.7 |
| `--part2` | Stitching; limiar do RANSAC; imagem de referência; força bruta vs FLANN | 5 |

Opções: `--tag NOME` (prefixo dos ficheiros de resultados), `--seqs SEQ1 SEQ2 ...` (só algumas
sequências do HPatches, para testar depressa) e `--methods M1 M2 ...` (só alguns métodos). Por omissão,
`--part1`, `--rotscale`, `--graf` e `--qualitative` correm os 6 métodos clássicos e os 3 aprendidos
(`SuperPoint+NN`, `SuperPoint+LightGlue`, `SIFT+LightGlue`).

Testes de sanidade (precisam das imagens GRAF): `python tests/test_sanity.py`

### Parte 2 (stitching)

```bash
python src/main.py --part2          # as três experiências abaixo
```

O pipeline é o do enunciado (§3.1): detetar, descrever, emparelhar, homografia com RANSAC entre imagens
adjacentes, composição das homografias até à imagem de referência, warp, canvas e média dos píxeis na
sobreposição. Não se usa o `cv2.Stitcher`. O comando faz três experiências, nos conjuntos `keble` e `proprio`:

| Experiência | Tabela (`results/tables/`) | Figuras e panoramas (`results/`) |
|---|---|---|
| Configurações (SIFT, ORB, KAZE, SuperPoint+NN, SuperPoint+LightGlue): matches, inliers, rácio de inliers e erro de reprojeção por par | `stitching_T8.csv` | `figures/stitching/<conjunto>/` (keypoints e inliers), `panoramas/<conjunto>/<método>.jpg` |
| Limiar do RANSAC (0,5 a 10 px), com os mesmos matches | `ransac_threshold.csv` | `figures/ransac_study/`, `panoramas/ransac/` |
| Imagem de referência (cada uma das 3 imagens): tamanho do canvas e ampliação de área | `reference.csv` | `figures/reference_study/`, `panoramas/reference/` |

Cada pasta de figuras tem um `legendas.md` com as legendas e os números da corrida. Os conjuntos de
imagens, os métodos, os limiares e o método dos dois estudos (SIFT) estão em `CFG["stitching"]`; os
parâmetros do RANSAC (3 px, 2000 iterações, confiança 0,995) em `CFG["ransac"]`. As fotografias com o lado
maior acima de 1000 px são reduzidas antes do stitching. `--methods` escolhe outras configurações.

Para um panorama avulso, com qualquer método e imagens:

```bash
python src/stitching.py data/Panorama/keble_a.jpg data/Panorama/keble_b.jpg data/Panorama/keble_c.jpg --method ORB --ref 1 --thr 3
```

Todos os parâmetros estão num único ficheiro, `src/config.py`. Os principais:

| Parâmetro | Valor |
|---|---|
| Keypoints por imagem | no máximo 2000 (os de maior resposta), em todos os métodos |
| Detetores e descritores | valores por omissão do OpenCV 4.13; FAST com threshold 10 e NMS |
| Matching | ratio test 0,8 + cross-check; L2 nos descritores float, Hamming nos binários |
| Match correto | erro de reprojeção com a homografia verdadeira < 3 px |
| RANSAC | `cv2.findHomography`, limiar 3 px, máximo de 2000 iterações, confiança 0,995 |
| Stitching | fotografias reduzidas a 1000 px de lado maior; imagem central como referência; média na sobreposição |
| Seed | 0 |
  
## 7. Estrutura

```
src/
├── main.py            ponto de entrada (um comando por experiência)
├── config.py          todos os parâmetros
├── interface.py       formato comum do resultado de uma imagem (Features)
├── datasets.py        leitura do HPatches, do GRAF e dos panoramas
├── detectors.py       SIFT, ORB, KAZE, FAST
├── descriptors.py     descritores e extração completa (detetar + descrever)
├── brief_own.py       BRIEF implementado (G I, G II, G III, G V)
├── learned.py         SuperPoint, SIFT do LightGlue, LightGlue e SuperGlue
├── matching.py        ratio test + cross-check; força bruta ou FLANN
├── methods.py         lista dos métodos; extrair e emparelhar
├── evaluation.py      métricas, corrida sobre os pares, tabelas-resumo
├── homography.py      homografia com RANSAC, cantos, composição de homografias
├── stitching.py       canvas, warp e média dos panoramas
├── visualization.py   figuras
├── studies/           rotação/escala sintética, FAST, BRIEF, figuras qualitativas, gráficos,
│                      stitching, limiar do RANSAC e imagem de referência
├── main.py            ponto de entrada
├── learned.py         SuperPoint, SIFT do LightGlue e matcher LightGlue
├── homography.py      homografia com RANSAC, erro de reprojeção, composição
└── stitching.py       canvas, warp e composição dos panoramas
tests/test_sanity.py   testes de sanidade
results/               tables/, figures/, panoramas/
```
