# Parâmetros de todas as experiências    
import copy
from pathlib import Path 
import cv2
 
ROOT = Path(__file__).resolve().parents[1]

CFG = {
    "seed": 0,    

    "paths": {
        "hpatches": ROOT / "data" / "hpatches-sequences-release",
        "graf": ROOT / "data" / "graf",
        "panorama": ROOT / "data" / "Panorama",
        "tables": ROOT / "results" / "tables",
        "figures": ROOT / "results" / "figures",
        "panoramas": ROOT / "results" / "panoramas",
    },

    # Protocolo de avaliação  
    "eval": {
        "eps_px": 3.0,                # match correto se o erro com a H ground-truth for < 3 px
        "pairs_k": [2, 3, 4, 5, 6],   # pares 1->k de cada sequência HPatches
        "categories": ["i", "v"],     # iluminação / ponto de vista
        "corner_thr_px": [1, 3, 5],   # % de pares com corner error da homografia abaixo destes valores
    },

    # Matching: ratio test de Lowe + cross-check (emparelhamento 1-para-1)
    "matching": {
        "ratio": 0.8,
        # FLANN (só no estudo força bruta vs aproximado): KD-tree nos float, LSH nos binários
        "flann": {"kdtree_trees": 5, "lsh_tables": 6, "lsh_key_size": 12, "lsh_multi_probe": 1, "checks": 50},
    }, 
    
    "max_keypoints": 2000,

    # Detetores 
    "detectors": {
        "SIFT": {"nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6},
        "ORB": {"scaleFactor": 1.2, "nlevels": 8, "edgeThreshold": 31, "fastThreshold": 20},
        "KAZE": {"extended": False, "upright": False, "threshold": 0.001},
        "FAST": {"threshold": 10, "nonmaxSuppression": True, "type": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16},
    },

    # Descritores 
    "descriptors": {
        "BRIEF": {"bytes": 32, "use_orientation": False},
        "BRISK": {"thresh": 30, "octaves": 3, "patternScale": 1.0},
        "FREAK": {"orientationNormalized": True, "scaleNormalized": True, "patternScale": 22.0, "nOctaves": 4},
    },

    # BRIEF próprio
    "brief_own": {
        "patch_size": 48,    # S: o artigo não dá valor; 48 é o do BRIEF do OpenCV
        "n_bits": 256,       # BRIEF-32
        "blur_ksize": 9,     # janela 9x9
        "blur_sigma": 2.0,   
        "polar_radii": 8,    # G V: 8 anéis x 32 ângulos = 256 pontos  
    },

    # Teste sintético de rotação e escala; 0 graus e escala 1.0 são controlos
    "synthetic": {
        "base_image": ROOT / "data" / "graf" / "img1.ppm",
        "rotations_deg": list(range(0, 181, 15)),
        "scales": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0],
        "mask_feather_px": 10,   # largura da transição suave da máscara circular
    },

    # Estudo do FAST 
    "fast_study": {
        "thresholds": [10, 20, 40, 80],
        "nms": [True, False],
        "reference": "SIFT",
        "chance_shifts_px": [(20, 20), (-20, -20)],   # deslocamentos da H para o nível "ao acaso"
        "max_points_repeatability": 50_000,           # acima disto a repetibilidade não é calculada
    },

    # Figuras qualitativas 
    "qualitative": {"max_matches_drawn": 150},

    # RANSAC  
    "ransac": {"thr_px": 3.0, "max_iters": 2000, "confidence": 0.995},

    # Métodos aprendidos: SuperPoint e LightGlue
    "learned": {
        "device": None,    
        "resize": None,    
        # LightGlue: confiança mínima de um match e paragem antecipada (valores por omissão do pacote)
        "lightglue": {"filter_threshold": 0.1, "depth_confidence": 0.95, "width_confidence": 0.99},
        # SuperGlue (implementação do Hugging Face transformers): pesos "outdoor" e os valores do
        # SuperGlue original (match_threshold 0.2, 100 iterações de Sinkhorn)
        "superglue": {"weights": "magic-leap-community/superglue_outdoor", "match_threshold": 0.2,
                      "sinkhorn_iterations": 100},
    },

    # Parte 2  
    "stitching": {
        # conjuntos de imagens: uma pasta cada, imagens .jpg por ordem de nome (esquerda -> direita)
        "sets": {"keble": ROOT / "data" / "Panorama",
         "proprio1": ROOT / "data" / "panorama2" / "seq1",
         "proprio2": ROOT / "data" / "panorama2" / "seq2",
         "proprio3": ROOT / "data" / "panorama2" / "seq3"},
        "max_side": 1000,   # fotografias com o lado maior acima disto são reduzidas (px)
         
        "methods": ["SIFT", "ORB", "KAZE", "SuperPoint+NN", "SuperPoint+SuperGlue", "SuperPoint+LightGlue"],
        # estudo do limiar do RANSAC; os panoramas lado a lado são os do "study_method"
        "ransac_thresholds_px": [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
        "study_set": "keble",
        "study_method": "SIFT",       
        "matcher_study": {"methods": ["SIFT", "KAZE", "ORB"], "repeats": 5},
    },
}

def get_config() -> dict:  
    return copy.deepcopy(CFG)

def set_seeds(cfg: dict) -> None: 
    cv2.setRNGSeed(cfg["seed"])