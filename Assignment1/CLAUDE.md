# Assignment #1 — IAVC 2026/27 (Keypoints, Descriptors, Stitching)

Enunciado completo: `docs/Assignment_1_Python_Keypoints_Descriptors_Stitching.pdf`
Guia detalhado: `docs/Guia_Assignment1_IAVC.pdf`

## Regras do enunciado (não violar)
- Só Python. OpenCV `opencv-contrib-python==4.13.0.92` (NUNCA instalar também `opencv-python`).
- Correção de um match: erro de reprojeção com a H ground-truth < 3 px. Ratio test 0.8. Matching 1-para-1 (ratio + cross-check).
- Métricas obrigatórias por configuração e por par, com denominadores:
  PMR = N_putative/N_features; Precision = N_correct/N_putative;
  Matching score = N_correct/N_features; Recall = N_correct/N_correspondences.
- Também: nº keypoints, repetibilidade, inliers RANSAC, erro médio de reprojeção, corner error da homografia,
  tempos separados (deteção / descrição / matching), dimensão e memória dos descritores.
- Resultados SEMPRE separados por iluminação (`i_*`) e ponto de vista (`v_*`) do HPatches.
- Binários (ORB, BRIEF, BRISK, FREAK) → NORM_HAMMING; float (SIFT, KAZE, SuperPoint) → NORM_L2.
- Não usar cv2.Stitcher. Código modular, comentado, reprodutível. Parâmetros num único config.

## Dados
- HPatches em `data/hpatches-sequences-release/<seq>/{1..6}.ppm` e `H_1_k` (H leva img1 → imgk).
- `data/` está no .gitignore — nunca fazer commit de datasets nem pesos.

## Estrutura
src/detectors.py, descriptors.py (inclui brief_own.py), matching.py, evaluation.py,
homography.py, stitching.py, learned.py, visualization.py, main.py
results/tables/*.csv, results/figures/*

## Interface comum entre módulos
Cada método devolve: kps (lista cv2.KeyPoint), desc (np.ndarray ou None), matches (lista cv2.DMatch),
t_det, t_desc, t_match em ms. Não alterar esta interface sem combinar com o colega.

## Divisão
- Pessoa A (EU): clássicos (SIFT, ORB, KAZE, FAST+BRIEF/BRISK/FREAK), BRIEF próprio (≥2 estratégias, G I–V),
  estudo FAST (threshold, NMS), evaluation.py e corrida no HPatches completo.
- Pessoa B: SuperPoint, LightGlue, SIFT+LightGlue, homografia, stitching, estudo RANSAC, imagem de referência.

## Convenções
- Seeds fixas (np.random.default_rng(0), cv2.setRNGSeed(0)).
- Comentários e relatório em português.
