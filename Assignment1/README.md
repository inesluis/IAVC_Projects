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
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.13.0.92
```

Verificar:

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'KAZE_create'), hasattr(cv2, 'xfeatures2d'))"
# esperado: 4.13.0 True True
python -c "import torch, lightglue, transformers; print(torch.__version__, torch.cuda.is_available())"
```

**GPU no Windows.** `pip install torch` instala a versão só para CPU. Para usar uma GPU NVIDIA:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.
Para forçar o CPU, pôr `CFG["learned"]["device"] = "cpu"` em `src/config.py`.

## 3. Dados
   
| Dados | Origem | Pasta |
|---|---|---|
| HPatches, *sequences release* (116 sequências: 57 `i_*`, 59 `v_*`) | https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip | `data/hpatches-sequences-release/<seq>/` |
| GRAF (`img1.ppm`, …) | material da cadeira | `data/graf/` |
| Panorama keble (3 imagens) | material da cadeira | `data/Panorama/` |
| Panoramas próprios (3 sequências de 3 fotografias) | fotografias nossas, incluídas na entrega | `data/panorama2/seq1/`, `seq2/`, `seq3/` |

Depois de extrair o zip do HPatches, confirmar que não fica uma pasta repetida:

```
data/hpatches-sequences-release/
├── i_ajuntament/
│   ├── 1.ppm … 6.ppm
│   └── H_1_2 … H_1_6      # homografia verdadeira: imagem 1 -> imagem k
├── v_graffiti/
└── …
```

Nos panoramas, cada pasta é uma sequência; as imagens (`.jpg`) são lidas por ordem de nome, da
esquerda para a direita.

**Pesos pré-treinados** (não se treina nada). São descarregados automaticamente na primeira utilização e
não fazem parte da entrega:

| Modelo | Pesos | Implementação |
|---|---|---|
| SuperPoint (detetor + descritor) | `superpoint_v1` (Magic Leap) | [cvg/LightGlue](https://github.com/cvg/LightGlue) |
| LightGlue para SuperPoint e para SIFT (matcher) | `superpoint_lightglue`, `sift_lightglue` | [cvg/LightGlue](https://github.com/cvg/LightGlue) |
| SuperGlue (matcher) | `magic-leap-community/superglue_outdoor` | Hugging Face `transformers` 5.18 |

## 4. Reproduzir as experiências

Todos os comandos a partir da pasta `src/`. Para correr tudo de uma vez (cerca de 40 min com GPU):

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

Opções: `--tag NOME` (prefixo dos ficheiros), `--methods M1 M2 ...` (só alguns métodos) e
`--seqs SEQ1 SEQ2 ...` (só algumas sequências do HPatches, para um teste rápido, por exemplo
`python main.py --part1 --seqs i_nescafe v_graffiti --tag teste`).
  
Os resultados ficam em `results/tables/` (CSV), `results/figures/` (figuras, com as legendas em
`legendas.md`) e `results/panoramas/`. Os números do relatório são os dos ficheiros `final_*`.

## 5. Métodos avaliados

| Método | Detetor | Descritor | Matcher |
|---|---|---|---|
| SIFT, ORB, KAZE | o próprio | o próprio | força bruta, ratio test 0,8 + cross-check |
| FAST+BRIEF, FAST+BRISK, FAST+FREAK | FAST | BRIEF / BRISK / FREAK | força bruta (Hamming), ratio test 0,8 + cross-check |
| SuperPoint+NN | SuperPoint | SuperPoint | força bruta (L2), ratio test 0,8 + cross-check |
| SuperPoint+SuperGlue | SuperPoint | SuperPoint | SuperGlue (confiança ≥ 0,2) |
| SuperPoint+LightGlue | SuperPoint | SuperPoint | LightGlue (confiança ≥ 0,1) |
| SIFT+LightGlue | SIFT do pacote LightGlue | RootSIFT | LightGlue (confiança ≥ 0,1) |

## 6. Parâmetros

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
└── studies/           rotação/escala, FAST, BRIEF, figuras qualitativas, gráficos, stitching,
                       limiar do RANSAC e imagem de referência, força bruta vs FLANN 
results/               tables/, figures/, panoramas/
```
