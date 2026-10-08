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
- **Estado:** Implementada (incluindo o aquecimento, `evaluation._warm_up`)
- **Decisão:** `time.perf_counter()`, em ms. Deteção e descrição são medidas em chamadas separadas
  (`detect()` e depois `compute()`). A criação do objeto OpenCV não entra no tempo. O corte ao
  orçamento (D05) entra no tempo de deteção.
- **Justificação:** o enunciado pede tempos separados de deteção, descrição e matching.
- **Consequência a referir no relatório:** no SIFT e no KAZE, `compute()` reconstrói o espaço de
  escalas, por isso `t_det + t_desc` é maior do que um `detectAndCompute()` único.
- **Aquecimento (observado):** a primeira chamada ao ORB demorou 961 ms e as seguintes ~7 ms
  (inicialização interna do OpenCV). Por isso, `run_hpatches` corre cada pipeline uma vez no
  primeiro par, sem medir, antes de começar. Para métodos em GPU: `torch.cuda.synchronize()` antes
  de cada leitura do tempo (Pessoa B).
- **Como se reportam (`evaluation.evaluate_pair`):** `t_det_ms` e `t_desc_ms` = média por imagem
  (img1 e imgk); `t_match_ms` = por par (inclui as pesquisas 1→2 e 2→1 do cross-check);
  `t_total_ms` = 2·(t_det + t_desc) + t_match.
- **Nota:** as features da imagem 1 são calculadas uma vez por sequência e reutilizadas nos 5
  pares, por isso o tempo da img1 é o mesmo nas 5 linhas da sequência.

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

### D16 — Matching: ratio test + cross-check, força bruta
- **Estado:** Implementada (`matching.match_descriptors`, `matching.mutual_nn`)
- **Decisão:** para cada descritor da img1 procuram-se os 2 vizinhos mais próximos na imgk; aceita-se
  o match se `d1 < 0,8·d2` (ratio de Lowe) **e** se o melhor vizinho da imgk → img1 for o mesmo ponto
  (cross-check). Resultado: matches 1-para-1. Por omissão usa-se força bruta (`BFMatcher`, exato).
- **Justificação:** ratio 0,8 e emparelhamento 1-para-1 são o protocolo de referência do enunciado
  (§2.8). A força bruta é exata, por isso as diferenças entre métodos não são confundidas com erros da
  pesquisa aproximada.
- **Alternativas implementadas:**
  - `CFG["matching"]["backend"] = "flann"`: KD-trees para float, LSH para binários, `checks=50`. Para
    a comparação força bruta vs aproximado pedida no stitching (§3.1, passo 4).
  - `mutual_nn`: só vizinho mútuo, sem ratio (para descritores aprendidos, onde 0,8 pode ser
    demasiado restritivo).
- **Detalhe:** o cross-check usa só o 1.º vizinho no sentido 2→1, sem ratio (como o `crossCheck`
  do OpenCV). Pares com menos de 2 vizinhos são ignorados (o ratio não está definido).

### D17 — Correspondências ground-truth e repetibilidade
- **Estado:** Implementada (`evaluation.gt_correspondences`)
- **Decisão:**
  1. Zona comum: pontos da img1 cuja projeção H·p cai dentro da imgk, e pontos da imgk cuja projeção
     H⁻¹·p cai dentro da img1.
  2. Dois pontos podem corresponder se ‖H·p1 − pk‖ < 3 px (erro medido na imgk).
  3. `N_corresp` = **emparelhamento 1-para-1 máximo** desse grafo bipartido
     (`scipy.sparse.csgraph.maximum_bipartite_matching`).
  4. Repetibilidade = `N_corresp / min(n1_visíveis, nk_visíveis)`.
- **Justificação:** com o emparelhamento máximo, qualquer conjunto de matches corretos 1-para-1 cabe
  em `N_corresp`, o que garante **N_correct ≤ N_corresp** e **recall ≤ 1**. O greedy do `starter.py`
  (para cada ponto, o vizinho mais próximo; se já usado, desiste) pode subcontar e dar recall > 1.
- **Verificação:** com H = identidade e a mesma imagem, repetibilidade = precisão = recall = 1,0 nos 6
  métodos. Num subconjunto de 4 sequências (120 linhas): MS = PMR × precision, recall ≤ 1 e
  N_correct ≤ N_corresp em todas as linhas.
- **Limitação a referir:** a repetibilidade é calculada com os keypoints depois do `compute()`, por
  isso FAST+BRIEF/BRISK/FREAK têm valores ligeiramente diferentes (cada descritor remove pontos
  diferentes da borda). A repetibilidade "pura" do detetor FAST é medida no estudo do FAST (T6).

### D18 — Casos limite e agregação
- **Estado:** Implementada (`evaluation.evaluate_pair`, `evaluation.summarize`)
- **Casos limite:**
  - `N_putative = 0` → precisão 0 (o método falhou nesse par; com NaN a média ficaria inflacionada).
  - `N_corresp = 0` → recall e repetibilidade NaN (não há o que recuperar; excluídos da média).
  - `N_features = 0` → PMR e MS = 0.
- **Agregação:** média por par ("macro"), agrupada por pipeline × categoria (`i`/`v`/`rot`/`scale`)
  e por pipeline × categoria × k (curvas F1; no sintético `k` = nível). Os pares de controlo
  (coluna `control`) são excluídos das médias por categoria, mas aparecem nas curvas (D20). As contagens (denominadores) também são médias por par e
  aparecem ao lado das métricas na T2.
- **Robustez:** um erro num par/pipeline não pára a corrida: fica registado na coluna `error` e na
  tabela `errors`.

### D19 — Robustez discutida em 4 categorias + análise qualitativa
- **Estado:** Implementada (`main.py --robustness`, tabela `<tag>_robustness.csv`)
- **Decisão:** a robustez é avaliada em 4 categorias, cada uma com média por categoria e por nível:

  | Categoria | Dados | Nível (coluna `k`) |
  |---|---|---|
  | `i` iluminação | HPatches `i_*` (57 seq) | par 1→k, k = 2..6 |
  | `v` ponto de vista | HPatches `v_*` (59 seq) | par 1→k, k = 2..6 |
  | `rot` rotação | sintético a partir da GRAF img1 (D20) | ângulo (°) |
  | `scale` escala | sintético a partir da GRAF img1 (D20) | fator de escala |

  Mais uma análise qualitativa com o GRAF 1→2 e 1→4 (figuras de matches certos e errados, P08).
- **Justificação:** o professor pediu explicitamente a discussão de escala, rotação, iluminação e
  ponto de vista (§5 do enunciado). O HPatches só separa `i` e `v`: as `v_*` misturam perspetiva,
  rotação e escala, por isso não permitem isolar a rotação nem a escala.
- **Porque não usar o GRAF para a rotação/escala:** o GRAF é uma sequência de mudança de ponto de
  vista (perspetiva forte, com alguma rotação/escala misturadas). Só tem 2 pares e os valores não são
  controlados, o que não dá médias nem curvas com significado. Serve como exemplo qualitativo,
  como o enunciado sugere ("particularly for qualitative analysis").
- **Complemento real possível:** `v_bark` e `v_boat` (sequências de zoom + rotação do conjunto de
  Oxford, incluídas no HPatches), para confirmar em imagens reais o que o sintético mostra.

### D20 — Teste sintético de rotação e escala
- **Estado:** Implementada (`studies/rot_scale_study.py`, `CFG["synthetic"]`, `main.py --rotscale`)
- **Imagem base:** GRAF img1 (320×400, textura rica).
- **Níveis:** rotação 0°–180° em passos de 15° (13 níveis); escala 0,5; 0,6; 0,7; 0,8; 0,9; 1,0;
  1,25; 1,5; 1,75; 2,0 (10 níveis). 0° e 1,0 são **controlos**: entram nas curvas, mas não nas
  médias por categoria (seriam triviais e inflacionariam a média).
- **Rotação:** em torno do centro, na mesma tela. Antes de rodar, a imagem base fica só com o círculo
  inscrito, com transição suave (10 px) para um cinzento uniforme (a média da imagem). Assim a rotação
  não cria nem perde conteúdo, e a borda não gera keypoints falsos. A img1 do par de rotação é esta
  imagem mascarada.
- **Escala:** `cv2.resize` em torno do centro, na mesma tela: `INTER_AREA` para reduzir (evita
  aliasing), `INTER_LINEAR` para ampliar. Na redução, a margem é preenchida com o mesmo cinzento; na
  ampliação, a imagem é recortada. A H inclui a convenção de centros de pixel do `cv2.resize`
  (`x' = s·(x + 0,5) − 0,5 + offset`).
- **H ground-truth:** é a própria transformação aplicada (exata).
- **Verificação:** `warpPerspective(img1, H)` comparado com a imagem gerada dá um erro médio de
  0,0–0,34 níveis de cinzento (rotação e ampliação) e ≤ 2,3 níveis na redução (diferença entre a
  suavização do `INTER_AREA` e a interpolação linear, não um erro geométrico).
- **Correção feita:** a 1.ª versão usava fundos com cinzentos ligeiramente diferentes na máscara e
  na rotação, o que criava arestas ténues nos cantos da imagem rodada. Passou a usar-se um único
  valor de fundo.
- **Resultados preliminares (só para validar o teste; repetir na corrida oficial):** coerentes com
  a teoria. FAST+BRIEF: precisão 0,88 a 15°, 0,17 a 30° e 0 a partir de 45° (BRIEF sem orientação).
  FAST+BRIEF/BRISK/FREAK: 0,02–0,14 a 0,5× e a 2× (FAST sem escala). SIFT e KAZE ≥ 0,79 em todos os
  níveis. ORB 0,87–0,89 com rotações > 135°.
- **Limitações a referir:** é uma única imagem (planar, textura de graffiti); não há ruído nem
  mudança de iluminação; a média por categoria depende dos níveis escolhidos (por isso as curvas por
  nível são o resultado principal).

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
- **Incidente (2026-10-08, 11:26):** ao instalar o LightGlue no `cvc`, o pip trouxe `opencv-python`
  5.0.0.93, que se sobrepôs ao contrib (`cv2.__version__` passou a 5.0.0 e `cv2.KAZE_create` deixou de
  existir). Corrigido com `pip uninstall -y opencv-python opencv-contrib-python` +
  `pip install opencv-contrib-python==4.13.0.92`. Depois disso o LightGlue continua a importar.
  **Regra:** depois de instalar qualquer pacote, confirmar com
  `python -c "import cv2; print(cv2.__version__, hasattr(cv2,'KAZE_create'))"` → `4.13.0 True`.
  Aviso a registar: o torch indica que `torch.jit.script` não é suportado em Python 3.14 (só um
  aviso por agora; a Pessoa B deve confirmar que o LightGlue corre).

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
- **P05 (2026-10-08)** — **Métricas da homografia por implementar** em `evaluation.py`: inliers do
  RANSAC, rácio de inliers, erro médio de reprojeção, corner error e % de pares com erro ≤ 1/3/5 px
  (tabela T4). Dependem de `homography.estimate_homography` (Pessoa B), que ainda não existe. Quando
  existir: chamá-la em `evaluate_pair`, acrescentar as colunas e a T4 no `summarize`, e registar os
  limiares no config.
- **P06 (2026-10-08)** — **Pipelines aprendidos por registar** em `methods.py` (SuperPoint+NN,
  SuperPoint+LightGlue, SIFT+LightGlue): depende de `learned.py` (Pessoa B).
- **P07 (2026-10-08)** — `main.py`: existem `--part1`, `--rotscale` e `--robustness`. Faltam
  `--fast` e `--brief` (Pessoa A, próximos) e `--part2` (Pessoa B).
- **P08 (2026-10-08)** — Análise qualitativa GRAF 1→2 e 1→4 (figuras de keypoints e de matches
  certos/errados) por fazer: `visualization.py`.
