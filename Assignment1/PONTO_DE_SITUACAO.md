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
| Contrato comum: `Features`, `MatchResult`, `Pipeline` | `src/interface.py` | Os teus métodos têm de devolver isto |
| Parâmetros únicos (inclui `CFG["ransac"]`: 3 px, 2000 iterações, confiança 0,995) | `src/config.py` | Não ponhas valores fixos no código; lê de `cfg` |
| Leitura/escrita de imagens | `datasets.load_image`, `load_image_set`, `imread`, `imwrite` | **Usa estas, não `cv2.imread`**: no Windows falha sem aviso em caminhos com acentos |
| Métodos clássicos | `descriptors.extract_classic("SIFT", gray, cfg)` (também `"ORB"`, `"KAZE"`, `"FAST+BRIEF"`, `"FAST+BRISK"`, `"FAST+FREAK"`) | Para as ≥ 3 configurações do stitching |
| Matching | `matching.match_descriptors` (ratio 0,8 + cross-check), `mutual_nn`, `matches_to_points` | Para "SuperPoint + matching de descritores" e para o stitching |
| Registo de pipelines | `methods.register(Pipeline(...))` | Ao registares os teus, a avaliação corre-os sem mais código |
| Avaliação | `evaluation.py`, `python src/main.py --part1` | Métricas no HPatches, rotação/escala e GRAF para qualquer pipeline registado |
| Figuras | `visualization.plot_matches(..., good=máscara)`, `plot_keypoints`, `plot_curves`, `save_figure` | Para "visualizar keypoints e inliers finais": passa a máscara de inliers em `good` |

Exemplo de uso de um método clássico no stitching:

```python
from config import get_config, setup_env
from datasets import load_image
from descriptors import extract_classic
from matching import match_descriptors, matches_to_points

cfg = get_config(); setup_env(cfg)
bgr1, gray1 = load_image("data/Panorama/keble_a.jpg")
bgr2, gray2 = load_image("data/Panorama/keble_b.jpg")
f1, f2 = extract_classic("SIFT", gray1, cfg), extract_classic("SIFT", gray2, cfg)
mr = match_descriptors(f1, f2, cfg)              # mr.matches: lista de cv2.DMatch (1-para-1)
p1, p2 = matches_to_points(f1, f2, mr.matches)   # arrays (M, 2) para o findHomography
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
def extract_superpoint(gray, cfg) -> Features: ...     # desc float32 (N, 256), norm = cv2.NORM_L2
def extract_sift_lg(gray, cfg) -> Features: ...        # SIFT no formato que o LightGlue-SIFT espera
def match_lightglue(f1, f2, cfg) -> MatchResult: ...   # matches como cv2.DMatch (queryIdx -> img1)
```

Regras do contrato:

- `gray` é uma imagem `uint8 (H, W)`. Os keypoints devolvidos têm de estar nas **coordenadas da imagem
  original** (se redimensionares, reescala antes de devolver).
- `Features.kps` é uma lista de `cv2.KeyPoint`; os tensores de que o LightGlue precisa podem ir em
  `Features.meta` (a avaliação ignora esse campo).
- Tempos em ms com `time.perf_counter()`. Em GPU, `torch.cuda.synchronize()` antes de cada leitura.
  O SuperPoint não separa deteção de descrição: põe `t_det` = tempo total, `t_desc = 0.0` e
  `meta["joint_timing"] = True`. A avaliação já faz uma chamada de aquecimento por pipeline antes de medir.
- Parâmetros (nº máximo de pontos, etc.) vêm de `cfg`.

E o registo em `methods.py` (retira o comentário `PENDENTE` do topo):

```python
register(Pipeline("SuperPoint+NN", extract_superpoint, match_descriptors))
register(Pipeline("SuperPoint+LightGlue", extract_superpoint, match_lightglue))
register(Pipeline("SIFT+LightGlue", extract_sift_lg, match_lightglue))
```

Para correrem na avaliação: acrescenta os nomes a `methods.DEFAULT_PIPELINES`, ou usa
`python src/main.py --part1 --pipelines SuperPoint+LightGlue ...`.

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

Precisão média por categoria (limite de 2000 pontos, ratio 0,8 + cross-check, correto se erro < 3 px):

| Método | iluminação | ponto de vista | rotação | escala | tempo por par (ms) |
|---|---|---|---|---|---|
| SIFT | 0,735 | 0,747 | 0,995 | 0,974 | 336 |
| KAZE | 0,780 | 0,743 | 0,990 | 0,940 | 1291 |
| ORB | 0,696 | 0,686 | 0,922 | 0,943 | 41 |
| FAST+BRIEF | 0,777 | 0,602 | 0,091 | 0,570 | 23 |
| FAST+BRISK | 0,743 | 0,548 | 0,992 | 0,504 | 39 |
| FAST+FREAK | 0,676 | 0,544 | 0,986 | 0,551 | 60 |

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
