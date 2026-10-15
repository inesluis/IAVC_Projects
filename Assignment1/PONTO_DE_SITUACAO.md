# Ponto de situação — Assignment #1 (12 de outubro de 2026)

De: Carolina (Pessoa A: métodos clássicos + avaliação)
Para: Inês (Pessoa B: SuperPoint/LightGlue + homografia + stitching)

Este documento diz o que já está feito, o que podes usar e o que preciso de ti para fechar a Parte 1.
O detalhe de cada decisão está em [DECISOES.md](DECISOES.md) (as referências `Dxx` e `Pxx` são de lá).

## Resumo

- A minha parte que não depende de ti está feita e está no ramo `carolina` (o `main` ainda só tem a
  estrutura base).
- **Preciso de duas coisas tuas para terminar a Parte 1:** a função `estimate_homography` e os métodos
  aprendidos (SuperPoint, LightGlue) no formato combinado. Sem elas faltam a tabela da homografia e
  duas das seis configurações obrigatórias.
- Há três coisas a combinar antes da corrida final: resolução das imagens, limite de pontos e máquina.

## 1. Como obter o código

```bash
git fetch origin
git merge origin/carolina        # ou: git checkout carolina
```

Ambiente: conda env com Python 3.10+, `pip install -r requirements.txt`. Depois de instalares o
LightGlue (ou qualquer outro pacote), confirma sempre:

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'KAZE_create'))"   # esperado: 4.13.0 True
```

O LightGlue instala o `opencv-python` 5.0 por cima do contrib e o KAZE deixa de existir (aconteceu-me).
Correção: `pip uninstall -y opencv-python opencv-contrib-python` e depois
`pip install opencv-contrib-python==4.13.0.92`.

Os dados não estão no git. O HPatches fica em `data/hpatches-sequences-release/<seq>/` (sem a pasta
repetida que o zip cria); GRAF em `data/graf/`; panoramas em `data/Panorama/`.

## 2. O que já está feito e podes usar

| O quê | Onde | Para que te serve |
|---|---|---|
| Formato comum: classe `Features` (`kps`, `desc`, `norm`, `t_det`, `t_desc`, `method`, `n_detected`) | `src/interface.py` | Os teus métodos têm de devolver isto |
| Parâmetros únicos (inclui `CFG["ransac"]`: 3 px, 2000 iterações, confiança 0,995) | `src/config.py` | Não ponhas valores fixos no código; lê de `cfg` |
| Leitura/escrita de imagens | `datasets.load_image`, `load_image_set`, `imread`, `imwrite` | **Usa estas, não `cv2.imread`**: no Windows falha sem aviso em caminhos com acentos |
| Métodos clássicos | `methods.extract_features("SIFT", gray, cfg)` (também `"ORB"`, `"KAZE"`, `"FAST+BRIEF"`, `"FAST+BRISK"`, `"FAST+FREAK"`) | Para as ≥ 3 configurações do stitching |
| Matching | `methods.match_features(method, f1, f2, cfg)` (ratio 0,8 + cross-check), `matching.matched_points` | Para "SuperPoint + matching de descritores" e para o stitching |
| Lista de métodos | `methods.CLASSIC_METHODS` e as funções `extract_features` / `match_features` | Acrescentas os teus métodos aqui e a avaliação corre-os sem mais código |
| Avaliação | `evaluation.py`, `python src/main.py --part1` | Métricas no HPatches, rotação/escala e GRAF para qualquer método da lista |
| Figuras | `visualization.plot_matches(img1, img2, kps1, kps2, matches, good, ...)`, `plot_keypoints`, `plot_curves`, `save_figure` | Para "visualizar keypoints e inliers finais": passa a máscara de inliers em `good` |

O código foi simplificado em 12 de outubro (funções diretas, sem classes de pipeline nem registo).
Se tinhas visto uma versão anterior, os nomes mudaram; a correspondência está na entrada D27 do
DECISOES.md.

Exemplo de uso de um método clássico no stitching (testado):

```python
from config import get_config, set_seeds
from datasets import load_image
from matching import matched_points
from methods import extract_features, match_features

cfg = get_config()
set_seeds(cfg)
bgr1, gray1 = load_image("data/Panorama/keble_a.jpg")
bgr2, gray2 = load_image("data/Panorama/keble_b.jpg")
f1 = extract_features("SIFT", gray1, cfg)
f2 = extract_features("SIFT", gray2, cfg)
matches, t_match = match_features("SIFT", f1, f2, cfg)   # lista de cv2.DMatch (1-para-1) e tempo em ms
p1, p2 = matched_points(f1, f2, matches)                 # arrays (M, 2) para o findHomography
```

## 3. O que preciso de ti

### 3.1 `homography.estimate_homography` (o mais urgente)

```python
def estimate_homography(kps1, kps2, matches, cfg) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """RANSAC com cfg["ransac"] (thr_px, max_iters, confidence).
    Devolve (H 3x3 que leva img1 -> img2, máscara booleana de inliers, erro médio de reprojeção dos
    inliers em px). Com menos de 4 matches ou se falhar: (None, None, nan)."""
```

**Para quê:** a tabela T4 (inliers do RANSAC, rácio de inliers, erro de reprojeção, corner error, % de
pares com erro ≤ 1/3/5 px). São métricas obrigatórias (enunciado §2.7) e hoje não existem. Quando a
função existir, acrescento-as à avaliação (P05). Assim há uma única implementação do RANSAC para a
Parte 1 e para o stitching.

### 3.2 Métodos aprendidos em `learned.py`

Três funções no formato do contrato (os nomes são sugestão):

```python
def extract_superpoint(gray, cfg) -> Features: ...   # desc float32 (N, 256), norm = cv2.NORM_L2
def extract_sift_lg(gray, cfg) -> Features: ...      # SIFT no formato que o LightGlue-SIFT espera
def match_lightglue(f1, f2, cfg): ...                # devolve (matches, t_match); matches como cv2.DMatch
```

Regras do contrato:

- `gray` é uma imagem `uint8 (H, W)`. Os keypoints devolvidos têm de estar nas **coordenadas da imagem
  original** (se redimensionares, reescala antes de devolver).
- `Features.kps` é uma lista de `cv2.KeyPoint`. Se precisares de guardar os tensores para o LightGlue,
  acrescenta um campo à classe `Features` em `interface.py` (a avaliação só usa os campos atuais).
- Em cada `cv2.DMatch`, `queryIdx` é o índice na imagem 1 e `trainIdx` na imagem 2; matches 1-para-1.
- Tempos em ms com `time.perf_counter()`. Em GPU, `torch.cuda.synchronize()` antes de cada leitura.
  O SuperPoint não separa deteção de descrição: põe `t_det` = tempo total e `t_desc = 0.0` (e dizemos
  isso no relatório). A avaliação já faz uma chamada de aquecimento por método antes de medir.
- Parâmetros (nº máximo de pontos, etc.) vêm de `cfg`.

E a ligação em `methods.py`: uma lista com os nomes e um ramo em cada função (retira o comentário
`PENDENTE` do topo). Esqueleto:

```python
LEARNED_METHODS = ["SuperPoint+NN", "SuperPoint+LightGlue", "SIFT+LightGlue"]

def extract_features(method, gray, cfg):
    if method.startswith("SuperPoint"):
        return extract_superpoint(gray, cfg)
    if method == "SIFT+LightGlue":
        return extract_sift_lg(gray, cfg)
    return extract(method, gray, cfg)             # métodos clássicos

def match_features(method, f1, f2, cfg):
    if method.endswith("LightGlue"):
        return match_lightglue(f1, f2, cfg)
    return match_descriptors(f1, f2, cfg)         # ratio test + cross-check
```

Para correrem na avaliação: no `main.py`, o `--part1` e o `--rotscale` usam `CLASSIC_METHODS`; passa a
usar `CLASSIC_METHODS + LEARNED_METHODS`. O esqueleto acima não foi testado (não tenho as tuas funções).

**Para quê:** "SuperPoint + matching" e "SuperPoint + LightGlue" são configurações mínimas obrigatórias
(§4). Sem elas, as tabelas e os gráficos só têm os métodos clássicos (P06).

### 3.3 O resto da tua parte

- `requirements.txt` (torch, LightGlue) e a secção de instalação do README (`TODO (INÊS)`).
- `stitching.py`, o estudo do limiar do RANSAC, a comparação de imagens de referência e o `--part2`
  no `main.py`.

## 4. A combinar antes da corrida final

1. **Resolução:** o SuperPoint corre na imagem original? Os clássicos usam a original. Se tiveres de
   reduzir, temos de reduzir para todos.
2. **Limite de pontos:** 2000 por imagem em todos os métodos (`CFG["max_keypoints"]`); no SuperPoint,
   `max_num_keypoints=2000`.
3. **Máquina:** a corrida oficial (clássicos + aprendidos) tem de ser feita de uma vez, numa só máquina,
   para os tempos serem comparáveis. CPU ou GPU? O meu portátil (i7-12650H) tem o torch com CUDA.

## 5. Resultados que já existem (preliminares, só métodos clássicos)

As tabelas e as figuras estão fora do git (`results/`). Geram-se em cerca de 25 minutos:

```bash
python src/main.py --part1 --tag classic        # HPatches: 580 pares, ~15 min
python src/main.py --rotscale --tag classic     # rotação e escala sintéticas
python src/main.py --robustness --tag classic   # tabela iluminação / ponto de vista / rotação / escala
python src/main.py --graf --tag classic         # GRAF 1->2 e 1->3, por par
python src/main.py --qualitative                # figuras de keypoints e matches
python src/main.py --fast --tag classic         # estudo do FAST, ~7 min
python src/main.py --brief --tag classic        # BRIEF próprio, ~2 min
python src/main.py --curves --tag classic       # gráficos a partir dos CSV
```

Precisão média por categoria (limite de 2000 pontos, ratio 0,8 + cross-check, correto se erro < 3 px).
Valores da corrida de 12 de outubro, com todos os detetores nos parâmetros por omissão do OpenCV (o
FAST passou de threshold 20 para 10 nesse dia; detalhes na D28 do DECISOES.md).

| Método | iluminação | ponto de vista | rotação | escala | tempo por par (ms) |
|---|---|---|---|---|---|
| SIFT | 0,735 | 0,747 | 0,995 | 0,974 | 328 |
| KAZE | 0,780 | 0,743 | 0,990 | 0,940 | 1262 |
| ORB | 0,696 | 0,686 | 0,922 | 0,943 | 43 |
| FAST+BRIEF | 0,776 | 0,597 | 0,080 | 0,551 | 33 |
| FAST+BRISK | 0,739 | 0,546 | 0,987 | 0,494 | 51 |
| FAST+FREAK | 0,674 | 0,542 | 0,983 | 0,568 | 73 |

Os tempos variaram até ±50% entre corridas na mesma máquina (carga ou aquecimento); para a corrida
oficial convém a máquina ligada à corrente e sem outros programas.

O que está feito além disto: estudo do FAST (threshold × NMS contra SIFT, D24), BRIEF próprio com 4
estratégias de amostragem (D25) e os gráficos de curvas e de exatidão contra tempo (D26).

Três escolhas que afetam a tua parte e convém saberes:

- **Rotação e escala** são avaliadas num teste sintético (GRAF imagem 1 rodada/escalada), porque o
  HPatches só separa iluminação e ponto de vista (D19, D20). Os teus métodos entram nesse teste
  automaticamente.
- **A `img4.ppm` do GRAF chama-se "Imagem 3"** nas figuras e tabelas (só temos 3 imagens GRAF; D21).
- **O FAST sem NMS devolve confiança 0 em todos os pontos**: com limite de pontos, usa sempre o FAST com
  NMS (D24).

## 6. O que falta para a entrega

| O quê | Quem |
|---|---|
| `estimate_homography` e métodos aprendidos (secção 3) | Inês |
| Tabela T4 (homografia) na avaliação | Carolina, depois da 3.1 |
| Parte 2 completa (stitching, RANSAC, referência) | Inês |
| Corrida oficial final com todos os métodos | As duas (secção 4) |
| `tests/test_sanity.py` (existe só o esqueleto) | Carolina |
| Relatório: Introdução, métodos clássicos, setup e resultados da Parte 1 | Carolina |
| Relatório: métodos aprendidos, RANSAC, resultados da Parte 2 | Inês |
| README final, juntar os ramos no `main`, zip com resultados | As duas |

Dúvidas ainda por confirmar com o professor: que imagens usar no stitching (P02), se os métodos
aprendidos podem usar um subconjunto do HPatches (P03), data de entrega, língua e limite de páginas (P04).

---

## 7. Resposta da Inês (14 de outubro de 2026)

O detalhe está no DECISOES.md (D29–D38) e no VERIFICACOES.md (secção 3).

### 7.1 O que está feito (ramo `Ines`, commit `4615590`)

```bash
git fetch origin
git merge origin/Ines
```

| Pedido | Estado | Onde |
|---|---|---|
| 3.1 `estimate_homography` | Feito, com a assinatura e o retorno que pediste. Seed reposta em cada chamada (resultado determinístico) | `src/homography.py`, D32 |
| 3.2 Métodos aprendidos | Feito: `extract_superpoint`, `extract_sift_lg`, `match_lightglue`; ligados em `methods.py` (`LEARNED_METHODS`, `ALL_METHODS`) | `src/learned.py`, D29–D31 |
| 3.2 `main.py` com os aprendidos | Feito: `--part1`, `--rotscale`, `--graf` e `--qualitative` correm os 9 métodos por omissão | `src/main.py` |
| 3.3 `requirements.txt` e README | Feito | |
| 3.3 Parte 2 | Feito: stitching, estudo do limiar, estudo da referência, `--part2` | D33–D36 |

Testei os aprendidos na tua avaliação (4 sequências + sintético): correm sem erros (D37).

### 7.2 Respostas à secção 4

1. **Resolução:** a original (`resize = None`), como nos clássicos.
2. **Limite de pontos:** 2000 (`max_num_keypoints`). É um teto: o SuperPoint deteta menos em várias
   imagens.
3. **Máquina:** proposta na D38. O meu PC (Ryzen 7 8700F + RTX 5060 Ti, desktop) evita as variações de
   tempo do portátil (D28), mas o teu também serve. O importante é ser tudo numa só máquina.

### 7.3 O que precisas de saber para as tuas tabelas e o teu texto

- **T4 (P05):** já podes chamar `estimate_homography` em `evaluate_pair`. Devolve `(None, None, nan)`
  com menos de 4 matches.
- **Tempos:** nos aprendidos, `t_desc = 0` e `t_det` = tempo total da rede. Os aprendidos correm em GPU
  e os clássicos em CPU: dizer isto junto da T5.
- **Protocolo de matching (§2.8):** o LightGlue não usa ratio test. O match putativo é um par com
  confiança ≥ 0,1 e `DMatch.distance = 1 − confiança`. O SuperPoint+NN usa o teu ratio 0,8 +
  cross-check (D31).
- **SIFT+LightGlue** usa o SIFT do pacote (RootSIFT, outros parâmetros), não o `cv2.SIFT` da
  configuração "SIFT" (D30).
- **Robustez:** o SuperPoint não é invariante à rotação: no teste sintético cai para ≤ 0,06 de
  precisão a partir de 60° (D37). Vai aparecer na tabela de robustez e nas curvas.
- **Ficheiros teus que alterei:** `interface.py` (campo `raw`), `methods.py` e `main.py` (ligação dos
  aprendidos e `--part2`), `visualization.py` (nova `plot_image_grid`), `curves.py` (mais 2
  marcadores, para 9 métodos), `config.py` (secções `learned`, `ransac` e `stitching`).
- **Numeração no DECISOES.md:** usei D29–D39 e P11–P13. As tuas próximas entradas começam em D40 e
  P14 (a D39, força bruta vs FLANN, foi acrescentada em 15/10).

### 7.4 O que falta

| O quê | Quem |
|---|---|
| T4 em `evaluation.py` (P05) | Carolina |
| Combinar a máquina e fazer a corrida oficial (D38) | As duas |
| ~~Força bruta vs FLANN no stitching (P12)~~ Feito em 15/10 (D39) | Inês |
| Relatório: métodos aprendidos, RANSAC, resultados da Parte 2 | Inês |
