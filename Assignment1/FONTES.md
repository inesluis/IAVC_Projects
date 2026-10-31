# Fontes — Assignment #1

Registo das fontes em que o trabalho se baseia, para a bibliografia do relatório. Atualizado em
12 de outubro de 2026; parte da Pessoa B acrescentada em 14 de outubro de 2026.

Cada fonte tem um **estado**:

- **Lida**: foi aberta e lida (pelo menos as secções indicadas) durante este trabalho.
- **Por ler**: tem de ser citada, mas o que o código e as notas dizem sobre ela vem de conhecimento
  geral, não de leitura feita neste trabalho. Ler as secções relevantes antes de escrever o relatório.
- **Consultada**: documentação técnica verificada.

As referências dos artigos estão escritas como no enunciado (secção 10), exceto onde indicado.

---

## 1. Material da cadeira

| Fonte | Estado | Usada para |
|---|---|---|
| Enunciado: "Assignment #1 — Keypoint Detectors, Feature Descriptors and Applications", Artificial Intelligence for (Advanced) Computer Vision, edição 2026/27 (`docs/Assignment_1_Python_Keypoints_Descriptors_Stitching.pdf`) | Lida (completo) | Tudo: métodos a avaliar (§2, §4), protocolo e métricas (§2.7, §2.8), funções a usar (§11), estrutura e reprodutibilidade (§6, §7) |
| Guia de estudo "Assignment #1 — Guia Completo", versão de 6 de outubro de 2026 (`docs/Guia_Assignment1_IAVC.pdf`) e o seu código de arranque (`docs/starter.py`) | Lida (completo) | Plano e divisão de trabalho; ponto de partida para as definições de repetibilidade e de correspondências ground-truth (depois alteradas, D17); origem do threshold 20 do FAST usado até 12 de outubro (D09) |
| Notebooks de exemplo dos docentes: `CVC_Sift.ipynb`, `CVC_LightGlue.ipynb`, `CVC_SuperPoint+SuperGlue.ipynb` (`códigos exemplo 2026/`) | **Não abertos** | Não foram a base do código (ver VERIFICACOES.md, 2.5). Só conhecemos o resumo que o guia faz deles |

## 2. Artigos

### 2.1 Lidos neste trabalho

| Referência | Secções lidas | Usada para |
|---|---|---|
| M. Calonder, V. Lepetit, C. Strecha, P. Fua, "BRIEF: Binary Robust Independent Elementary Features," ECCV 2010, pp. 778–792. DOI: 10.1007/978-3-642-15561-1_56 *(não está na lista do enunciado; é o artigo original; nº do volume LNCS por confirmar no DOI)* | §3 (Method), §3.1, §3.2; partes do §4 (detetor usado, matching, sensibilidade à orientação) e §5 | BRIEF próprio: definição do teste, suavização, as 5 geometrias de amostragem e os seus parâmetros (D25; VERIFICACOES.md, secção 1). Precedente para o teste de rotação sintética (imagem rodada de 0 a 180°) |
| M. Calonder, V. Lepetit, M. Özuysal, T. Trzcinski, C. Strecha, P. Fua, "BRIEF: Computing a Local Binary Descriptor Very Fast," IEEE TPAMI, 34(7), pp. 1281–1298, 2012. DOI: 10.1109/TPAMI.2011.222 | §III (Method), §III-A, §III-B; títulos e passagens do §IV | Diferenças em relação à versão de 2010: janela 7×7, filtro de caixa, G V = transformada Census, BRIEF-32 como omissão, variantes {U, O, S, D}-BRIEF |

### 2.2 Por ler (citar no relatório; confirmar as afirmações na fonte)

| Referência | Método / uso no trabalho |
|---|---|
| D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV, 60(2), pp. 91–110, 2004 | SIFT (baseline); ratio test |
| E. Rosten, T. Drummond, "Machine Learning for High-Speed Corner Detection," ECCV 2006 | FAST |
| E. Rosten, R. Porter, T. Drummond, "Faster and Better: A Machine Learning Approach to Corner Detection," IEEE TPAMI, 32, pp. 105–119, 2010 | FAST (estudo do threshold e da NMS, D24) |
| P. F. Alcantarilla, A. Bartoli, A. J. Davison, "KAZE Features," ECCV 2012 | KAZE |
| E. Rublee, V. Rabaud, K. Konolige, G. Bradski, "ORB: An Efficient Alternative to SIFT or SURF," ICCV 2011 | ORB |
| S. Leutenegger, M. Chli, R. Siegwart, "BRISK: Binary Robust Invariant Scalable Keypoints," ICCV 2011 | BRISK |
| A. Alahi, R. Ortiz, P. Vandergheynst, "FREAK: Fast Retina Keypoint," CVPR 2012 | FREAK |
| J. Heinly, E. Dunn, J.-M. Frahm, "Comparative Evaluation of Binary Features," ECCV 2012, LNCS 7573, pp. 759–773 | Protocolo e métricas: PMR, precisão, matching score, recall; 3 px; ratio 0,8 (o enunciado, §2.8, resume este protocolo; usámos o resumo do enunciado) |
| V. Balntas, K. Lenc, A. Vedaldi, K. Mikolajczyk, "HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors," CVPR 2017 | Dataset HPatches |
| H. Bay, A. Ess, T. Tuytelaars, L. Van Gool, "SURF: Speeded Up Robust Features," CVIU, 110(3), pp. 346–359, 2008 | SURF: não avaliado (indisponível no OpenCV instalado, D14); citar ao justificar |
| R. Szeliski, "Computer Vision: Algorithms and Applications," 2.ª ed., Springer, 2022 | Referência geral (RANSAC, homografias, stitching) |
| D. DeTone, T. Malisiewicz, A. Rabinovich, "SuperPoint: Self-Supervised Interest Point Detection and Description," CVPR Workshops, 2018 | SuperPoint (detetor + descritor) nas configurações SuperPoint+NN e SuperPoint+LightGlue; pesos `superpoint_v1.pth` (D29) |
| P. Sarlin, D. DeTone, T. Malisiewicz, A. Rabinovich, "SuperGlue: Learning Feature Matching with Graph Neural Networks," CVPR 2020 | Não avaliado (não está na lista mínima do §4); citar ao descrever os matchers aprendidos e como antecessor do LightGlue |
| P. Lindenberger, P.-E. Sarlin, M. Pollefeys, "LightGlue: Local Feature Matching at Light Speed," ICCV 2023 *(não está na lista do enunciado; indicada no guia)* | Matcher LightGlue e implementação usada (pacote cvg/LightGlue); pesos `superpoint_lightglue` e `sift_lightglue` (D29–D31) |

### 2.3 Referências que provavelmente fazem falta (escritas de memória; confirmar antes de usar)

| Referência | Porquê |
|---|---|
| K. Mikolajczyk, T. Tuytelaars, C. Schmid, A. Zisserman, J. Matas, F. Schaffalitzky, T. Kadir, L. Van Gool, "A Comparison of Affine Region Detectors," IJCV, 65, pp. 43–72, 2005 | Origem das sequências GRAF (e de `v_bark`, `v_boat`, `v_wall`, `v_graffiti`); definição clássica de repetibilidade |
| C. Schmid, R. Mohr, C. Bauckhage, "Evaluation of Interest Point Detectors," IJCV, 37(2), pp. 151–172, 2000 | Definição de repetibilidade de detetores |
| M. A. Fischler, R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," Communications of the ACM, 24(6), pp. 381–395, 1981 | Artigo original do RANSAC; o relatório tem de explicar o princípio (§3.2, D32) |
| R. Hartley, A. Zisserman, "Multiple View Geometry in Computer Vision," 2.ª ed., Cambridge University Press, 2004 | Homografias (8 graus de liberdade, 4 correspondências) e estimação robusta; alternativa ou complemento ao Szeliski |

## 3. Documentação técnica

O site docs.opencv.org não permitiu abrir as páginas automaticamente; os endereços e o conteúdo abaixo
foram confirmados pelos resultados de pesquisa no próprio site e, para os valores, pela versão
instalada (VERIFICACOES.md, secção 2.2).

| Página | Endereço | Confirmado |
|---|---|---|
| OpenCV 4.13.0, cv::SIFT Class Reference | https://docs.opencv.org/4.13.0/d7/d60/classcv_1_1SIFT.html | Classe e algoritmo de Lowe |
| OpenCV, cv::FastFeatureDetector Class Reference | https://docs.opencv.org/3.4/df/d74/classcv_1_1FastFeatureDetector.html | `create(threshold = 10, nonmaxSuppression = true, type = TYPE_9_16)` |
| OpenCV 4.x, tutorial "FAST Algorithm for Corner Detection" | https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html | Uso do threshold e da NMS em Python |
| OpenCV, cv::xfeatures2d::BriefDescriptorExtractor Class Reference | https://docs.opencv.org/3.4/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html | `bytes` = 16, 32 (omissão) ou 64; baseado em Calonder et al., ECCV 2010 |
| OpenCV 4.x, cv::xfeatures2d::FREAK Class Reference | https://docs.opencv.org/4.x/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html | Parâmetros de `create` |
| OpenCV, cv::BFMatcher Class Reference | https://docs.opencv.org/3.4.20/d3/da1/classcv_1_1BFMatcher.html | L1/L2 para SIFT/SURF; `NORM_HAMMING` para ORB, BRISK e BRIEF; `crossCheck` |
| OpenCV 4.13.0, tutorial "Feature Matching" | https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html | BFMatcher com `NORM_HAMMING` para ORB; ratio test |
| OpenCV, cv::Feature2D Class Reference | https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html | Interface comum `detect` / `compute` |
| OpenCV 4.x, namespace cv::xfeatures2d | https://docs.opencv.org/4.x/d3/df6/namespacecv_1_1xfeatures2d.html | Módulo contrib (BRIEF, FREAK, SURF) |

Software dos métodos aprendidos:

| Fonte | Endereço | Confirmado |
|---|---|---|
| Repositório cvg/LightGlue (código e pesos, release `v0.1_arxiv`) | https://github.com/cvg/LightGlue | Código do pacote instalado lido em 2026-10-14: `default_conf` do SuperPoint, do SIFT e do LightGlue (D30); keypoints do SuperPoint em píxeis inteiros (`superpoint.py`, linha 210); nomes dos ficheiros de pesos na cache |

Por confirmar (endereços não verificados): páginas de `cv::findHomography` e `cv::warpPerspective`
em docs.opencv.org (usadas na Parte 2); páginas de `cv::ORB`, `cv::KAZE` e `cv::BRISK` em
docs.opencv.org; páginas do SciPy para `scipy.spatial.cKDTree.sparse_distance_matrix` e
`scipy.sparse.csgraph.maximum_bipartite_matching` (usadas nas correspondências ground-truth, D17 e D24).

## 4. Dados

| Dados | Origem | Uso |
|---|---|---|
| HPatches, "hpatches-sequences-release" (116 sequências: 57 `i_*`, 59 `v_*`) | https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip (endereço indicado no guia); artigo de Balntas et al., CVPR 2017 | Avaliação quantitativa (iluminação e ponto de vista) |
| GRAF: `img1.ppm`, `img2.ppm`, `img4.ppm`, `H1to2p.txt`, `H1to4p.txt` | Material da cadeira (conjunto de Oxford; ver 2.3) | Teste sintético de rotação/escala, figuras qualitativas, métricas por par |
| Panorama: `keble_a.jpg`, `keble_b.jpg`, `keble_c.jpg` (720×568) | Material da cadeira | Parte 2: conjunto `keble` (D33) |
| `vehicle-seqs/` | Material da cadeira | Não usado |
| Panorama próprio: `img1.jpg`, `img2.jpg`, `img3.jpg` (3024×4032, reduzidas a 750×1000) em `data/panorama2/` | Fotografias nossas | Parte 2: conjunto `proprio` (D33) |
| Pesos pré-treinados: `superpoint_v1.pth`, `superpoint_lightglue_v0-1_arxiv.pth`, `sift_lightglue_v0-1_arxiv.pth` | Descarregados pelo pacote cvg/LightGlue (release `v0.1_arxiv`; o SuperPoint é da Magic Leap) | Métodos aprendidos; não vão na entrega (D15) |

## 5. Software

| Software | Versão | Notas |
|---|---|---|
| Python | 3.14.7 (conda env `cvc`) | O enunciado pede 3.10 ou superior |
| opencv-contrib-python | 4.13.0.92 | Versão indicada no enunciado; SURF indisponível (non-free) |
| NumPy | 2.5.2 | |
| SciPy | 1.18.1 | |
| Matplotlib | 3.11.2 | |
| pandas | 3.0.6 | |
| PyTorch | 2.14.1+cu126 | Instalado no `cvc`; os métodos aprendidos não foram corridos neste ambiente |
| CPU | Intel Core i7-12650H (16 threads) | Indicar junto aos tempos |

Ambiente da Pessoa B (PC onde correram a Parte 2 e o teste da D37):

| Software / hardware | Versão | Notas |
|---|---|---|
| Python | 3.11.9 (venv) | |
| opencv-contrib-python | 4.13.0.92 | Sem `opencv-python` |
| PyTorch | 2.11.0+cu128 | CUDA 12.8 |
| LightGlue | GitHub cvg/LightGlue (versão declarada 0.0) | Pesos `v0.1_arxiv` |
| kornia | 0.8.3 | Dependência do LightGlue |
| NumPy / SciPy / pandas / Matplotlib | 2.4.6 / 1.17.1 / 3.0.6 / 3.11.2 | |
| CPU | AMD Ryzen 7 8700F (8 núcleos) | |
| GPU | NVIDIA GeForce RTX 5060 Ti | Usada pelos métodos aprendidos (`CFG["learned"]["device"] = None`) |

Os dois ambientes têm versões diferentes; a corrida oficial faz-se num só (D38).

## 6. Ferramentas de IA

O código, a documentação e as verificações deste repositório foram desenvolvidos com a assistência do
Claude Code (Anthropic), modelo Claude Opus 5.5, em outubro de 2026. Os commits identificam essa
coautoria. Declarar no relatório de acordo com as regras da cadeira sobre o uso de IA (a confirmar).

## 7. Afirmações feitas de memória, a confirmar nas fontes antes de irem para o relatório

Estas afirmações aparecem no DECISOES.md ou foram usadas para interpretar resultados. São
consistentes com o que se mediu, mas não foram lidas nos artigos durante este trabalho.

| Afirmação | Onde aparece | Fonte onde confirmar |
|---|---|---|
| O score do FAST (`KeyPoint.response`) é o maior threshold para o qual o píxel ainda é canto | D24 | Rosten et al. (TPAMI 2010); código do OpenCV. Medido: com NMS, o mínimo de `response` é igual ao threshold |
| O ORB ordena os pontos pelo score de Harris e calcula a orientação pelo centróide de intensidade | D21, guia | Rublee et al. (ICCV 2011). Medido: `scoreType` = Harris na versão instalada |
| O KAZE usa difusão não linear em vez de suavização gaussiana | guia | Alcantarilla et al. (ECCV 2012) |
| O SIFT/KAZE reconstroem o espaço de escalas em `compute()`, por isso t_det + t_desc > `detectAndCompute` | D06 | Comportamento da implementação; pode medir-se diretamente |
| O ratio test com 0,8 vem de Lowe | D16, guia | Lowe (IJCV 2004); o enunciado dá 0,8 como protocolo de referência (§2.8) |
| O `maximum_bipartite_matching` do SciPy usa o algoritmo de Hopcroft–Karp | D24 | Documentação do SciPy |
| O BRIEF do OpenCV usa filtro de caixa com imagens integrais, patch 48 e kernel 9 | D25 | Código-fonte do OpenCV (`brief.cpp`). Medido: margem de 28 px = 48/2 + 9/2 |
| As sequências GRAF, `v_bark`, `v_boat` vêm do conjunto de Oxford (zoom + rotação, ponto de vista) | D19 | Mikolajczyk et al. (IJCV 2005) |
| O SuperPoint é treinado por adaptação homográfica e não é invariante a rotações grandes | D37 | DeTone et al. (CVPRW 2018). Medido: precisão ≤ 0,06 a partir de 60° com LightGlue (D37); a causa no treino falta confirmar no artigo |
| O LightGlue pára mais cedo (`depth_confidence`) e descarta pontos sem par (`width_confidence`) nos pares fáceis | D30 | Lindenberger et al. (ICCV 2023); o enunciado (§2.6) descreve o mesmo |
| O RANSAC do `cv2.findHomography` ajusta o nº de iterações à proporção de inliers e à confiança (até `maxIters`) e, no fim, reestima H com todos os inliers | D32 | Documentação de `cv::findHomography`; Fischler & Bolles (1981) para o princípio |
| O ORB não refina a posição dos keypoints ao sub-píxel | D34 | Código do OpenCV (`orb.cpp`). Medido: só 26% das coordenadas inteiras (as do nível 0); as restantes são posições de níveis reduzidos da pirâmide |
