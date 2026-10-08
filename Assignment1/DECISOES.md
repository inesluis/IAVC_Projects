# Registo de decisões — Assignment #1

Cada decisão de implementação ou de protocolo fica registada aqui, com a justificação, as
alternativas consideradas e o que deve aparecer no relatório. O código refere estas entradas
como `Dxx` nos comentários.

Legenda de estado: **Implementada** · **Acordada** (decidida, ainda por implementar) · **Pendente** (falta decidir/confirmar)

---

## Protocolo e interface

### D01 — Interface comum entre módulos
- **Estado:** Implementada (`src/interface.py`)
- **Decisão:** cada extrator devolve `Features` (kps, desc, norm, t_det, t_desc, method, meta) e cada
  matcher devolve `MatchResult` (matches, t_match, method, meta). Uma configuração avaliável é um
  `Pipeline` = extrator + matcher.
- **Regras:** keypoints em coordenadas da imagem original; `desc` float32 (N, D) para descritores
  float e uint8 empacotado (N, bits/8) para binários; `queryIdx` → imagem 1, `trainIdx` → imagem 2;
  matches 1-para-1 e putativos (antes do RANSAC); tempos em ms.
- **Justificação:** permite que `evaluation.py` (Pessoa A) avalie os métodos da Pessoa B e que
  `stitching.py` (Pessoa B) use os da Pessoa A sem mexer no código um do outro. Separar extrator de
  matcher obriga a distinguir desempenho do detetor/descritor do desempenho do matcher, como pede o
  enunciado (§2.6, §4). Ex.: "SuperPoint+NN" e "SuperPoint+LightGlue" partilham o extrator.
- **Alternativa:** tuplos soltos (como no `starter.py`). Rejeitada: fácil de trocar a ordem dos
  campos e não há lugar para metadados (ex.: tensores do LightGlue).

### D02 — Denominadores das métricas
- **Estado:** Acordada (implementar em `evaluation.py`)
- **Decisão:** `#N_features` = todos os keypoints da imagem 1 (depois do `compute()`).
  `#N_correspondences` = keypoints da imagem 1 visíveis na imagem k que, projetados com a H GT,
  têm um keypoint da imagem k a < 3 px, emparelhados 1-para-1. A repetibilidade usa só os pontos
  na zona comum: `N_corresp / min(n1_visíveis, nk_visíveis)`.
- **Justificação:** o enunciado (§2.8) deixa a definição aos alunos mas exige que seja reportada.
  Usar todos os pontos em `N_features` segue a definição literal de PMR e matching score
  (Heinly et al.). A repetibilidade é uma medida só do detetor e, por isso, só faz sentido na zona
  onde o ponto pode reaparecer.
- **No relatório:** escrever as duas definições e reportar os denominadores em todas as tabelas.

### D03 — Uma única implementação do RANSAC
- **Estado:** Acordada (Pessoa B, `homography.py`)
- **Decisão:** `evaluation.py` usa o `estimate_homography` de `homography.py`. Parâmetros em
  `CFG["ransac"]`: limiar 3 px, 2000 iterações, confiança 0,995.
- **Justificação:** os inliers e o corner error da Parte 1 e o stitching da Parte 2 ficam calculados
  exatamente da mesma forma.

### D04 — Localização e dimensão do HPatches
- **Estado:** Implementada
- **Decisão:** o zip extraía para `data/hpatches-sequences-release/hpatches-sequences-release/`.
  A pasta interior subiu um nível; o caminho é `data/hpatches-sequences-release/<seq>/`.
- **Verificado:** 116 sequências (57 `i_*` + 59 `v_*`), ou seja 580 pares 1→k (k = 2..6). As
  imagens têm tamanhos diferentes entre sequências (ex.: 480×640 em `i_ajuntament`, 640×800 em
  `v_graffiti`).
- **Decisão:** usar o HPatches completo para os métodos clássicos. Para os aprendidos em CPU decide-se
  depois (ver P03).

### D05 — Orçamento comum de 2000 keypoints
- **Estado:** Implementada (`detectors.cap_keypoints`, `CFG["max_keypoints"]`)
- **Decisão:** todos os detetores ficam limitados aos 2000 pontos com maior `response`, com
  ordenação estável (resultado determinístico). Com `max_keypoints=None` não há limite.
- **Justificação:** comparação justa (o guia, §6.4, e o enunciado, §5: "um número grande de
  keypoints não indica um bom método"). Sem limite, o FAST com threshold baixo produz dezenas de
  milhares de pontos e domina o PMR e os tempos.
- **Nota:** é um teto, não um alvo. Alguns métodos detetam menos (ex.: KAZE: 490 pontos em
  `i_ajuntament` 1). Isto faz parte do comportamento do detetor e deve ser discutido.
- **Alternativa:** usar o `nfeatures` de cada método. Só existe no SIFT e no ORB, por isso o corte é
  feito por nós, de forma igual para todos (exceção: ORB, ver D10).
- **Possível extra:** repetir a experiência sem limite e discutir a diferença.

### D06 — Medição de tempos
- **Estado:** Implementada (deteção/descrição) · Acordada (aquecimento)
- **Decisão:** `time.perf_counter()`, em ms. Deteção e descrição são medidas em chamadas separadas
  (`detect()` e depois `compute()`). A criação do objeto OpenCV não entra no tempo. O corte ao
  orçamento (D05) entra no tempo de deteção.
- **Justificação:** o enunciado pede tempos separados de deteção, descrição e matching.
- **Consequência a referir no relatório:** no SIFT e no KAZE, `compute()` reconstrói o espaço de
  escalas, por isso `t_det + t_desc` é maior do que um `detectAndCompute()` único.
- **Aquecimento (observado):** a primeira chamada ao ORB demorou 961 ms e as seguintes ~7 ms
  (inicialização interna do OpenCV). A avaliação vai fazer uma execução de aquecimento por método
  antes de medir. Para métodos em GPU: `torch.cuda.synchronize()` antes de cada leitura do tempo.

### D07 — Seeds fixas
- **Estado:** Implementada (`config.setup_env`)
- **Decisão:** `cv2.setRNGSeed(0)` e `np.random.default_rng(0)` no início de cada experiência.
- **Justificação:** reprodutibilidade (enunciado §7). Afeta o RANSAC e a amostragem de pares do
  BRIEF próprio.

## Implementação

### D08 — Leitura e escrita de imagens com caminhos acentuados
- **Estado:** Implementada (`datasets.imread`, `datasets.imwrite`)
- **Decisão:** ler com `np.fromfile` + `cv2.imdecode` e escrever com `cv2.imencode` + `tofile`.
- **Justificação:** no Windows, `cv2.imread` falha sem aviso (devolve `None`) em caminhos com
  caracteres não-ASCII, e o caminho do projeto tem "º" e "ê". Assim também funciona nos outros sistemas.
- **Regra:** nenhum módulo deve usar `cv2.imread`/`cv2.imwrite` diretamente.

### D09 — Parâmetros dos detetores
- **Estado:** Implementada (`CFG["detectors"]`)
- **Decisão:** valores por omissão do OpenCV 4.13, escritos explicitamente no config:
  - SIFT: 3 camadas/octava, contrastThreshold 0,04, edgeThreshold 10, σ 1,6
  - ORB: scaleFactor 1,2, 8 níveis, edgeThreshold 31, fastThreshold 20
  - KAZE: threshold 0,001, sem `extended` e sem `upright`
  - FAST: threshold 20, NMS ligada, tipo 9/16
- **Justificação:** os valores por omissão são os recomendados pelos autores/OpenCV. Afiná-los por
  método favoreceria uns métodos em relação a outros. O efeito dos parâmetros do FAST é estudado à
  parte (T6).

### D10 — ORB: `nfeatures` = orçamento
- **Estado:** Implementada
- **Decisão:** `ORB_create(nfeatures=max_keypoints)`; sem orçamento usa-se 100 000.
- **Justificação:** o valor por omissão do ORB é 500, o que o deixaria em desvantagem. O ORB
  distribui os pontos pelos níveis da pirâmide de acordo com `nfeatures`, por isso o limite tem de ser
  dado na criação (o corte posterior por si só não chega). O SIFT usa `nfeatures=0` + corte comum.

### D11 — Descritores e combinações válidas
- **Estado:** Implementada (`descriptors.py`)
- **Decisões:**
  - SIFT, KAZE e ORB só com o seu próprio detetor (dependem da escala/octava atribuídas por ele);
    combinações como "FAST+SIFT" dão erro.
  - BRIEF, BRISK e FREAK funcionam sobre keypoints FAST (configurações "FAST+X" do enunciado, §4).
  - BRIEF do OpenCV com 32 bytes (256 bits, valor do artigo) e sem orientação (BRIEF original).
  - BRISK e FREAK com os parâmetros por omissão.
  - `compute()` remove pontos perto da borda: usa-se sempre a lista devolvida.
    `meta["n_detected"]` guarda o número antes da filtragem (ex.: FAST 737 → BRIEF 583).
  - Se não sobrar nenhum ponto, `desc = None`.
  - Normas: L2 para SIFT/KAZE, Hamming para ORB/BRIEF/BRISK/FREAK.
- **Observação preliminar (1 par, não é resultado):** FAST+BRIEF cai para precisão 0,57 em
  `v_graffiti` 1→2 e 0,67 em GRAF 1→2, contra > 0,9 nos restantes métodos. É coerente com o BRIEF
  não ter orientação. Confirmar no HPatches completo e discutir.

### D12 — Threads do OpenCV
- **Estado:** Implementada (`CFG["num_threads"]`)
- **Decisão:** por omissão (`None`) o OpenCV usa todas as threads (16 nesta máquina).
- **Justificação:** é a forma normal de uso. Os tempos são reportados indicando o CPU. Se for
  preciso isolar o custo algorítmico, pode fixar-se `num_threads=1` e repetir.

## Ambiente e repositório

### D13 — Ambiente Python
- **Estado:** Implementada
- **Decisão:** desenvolvimento no conda env `cvc`: Python 3.14.7, opencv-contrib-python 4.13.0.92
  (`xfeatures2d` e `KAZE_create` verificados), torch 2.14.1+cu126.
  CPU: Intel Core i7-12650H (16 threads).
- **`requirements.txt`:** mínimo (numpy, scipy, matplotlib, pandas, psutil, opencv-contrib). Nunca
  `opencv-python` (partilha o módulo `cv2` e estraga o contrib). torch e LightGlue são acrescentados
  pela Pessoa B.
- **pandas:** instalado no `cvc` em 2026-10-08 (versão 3.0.6) para `evaluation.py`; o OpenCV ficou intacto
  (só opencv-contrib-python 4.13.0.92).

### D14 — SURF não avaliado
- **Estado:** Implementada (verificado)
- **Decisão:** o SURF não é avaliado. `cv2.xfeatures2d.SURF_create()` dá
  `error: (-213: The function/feature is not implemented)` no opencv-contrib-python 4.13.0.92
  (algoritmo non-free, desativado nas wheels do pip).
- **No relatório:** "SURF não avaliado: indisponível em opencv-contrib-python 4.13.0.92 (non-free)".
  O enunciado (§2.2) diz que é opcional.

### D15 — Dados e pesos fora do git
- **Estado:** Implementada (`.gitignore`)
- **Decisão:** ficam fora do git `data/`, `*.pth`, `__pycache__/`, `.ipynb_checkpoints/`, `venv/` e
  `.venv/`. O README explica como obter o HPatches.
- **Justificação:** enunciado §7 ("não incluir datasets grandes nem pesos").

---

## Pendentes / a confirmar

- **P01** — Perguntar ao professor se o BRIEF próprio é obrigatório e com quantas estratégias. Plano:
  fazer G I e G II no mínimo, idealmente G I–V.
- **P02** — Que imagens usar no stitching (há `data/Panorama/keble_*.jpg` e `data/vehicle-seqs/`).
- **P03** — HPatches completo ou subconjunto para SuperPoint/LightGlue em CPU (a GPU com CUDA
  parece disponível no `cvc`: torch cu126).
- **P04** — Data de entrega, língua e limite de páginas do relatório.
