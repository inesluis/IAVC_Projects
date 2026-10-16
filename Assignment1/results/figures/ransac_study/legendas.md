# Legendas — limiar do RANSAC

Geradas automaticamente com os números da corrida.

**inlier_ratio.png**

Rácio de inliers (inliers / matches putativos) em função do limiar do RANSAC, no conjunto keble: média dos 2 pares de imagens adjacentes. Os matches de cada método são os mesmos em todos os limiares; só muda o limiar. RANSAC com no máximo 2000 iterações e confiança 0,995.

**panoramas.jpg**

Panoramas do conjunto keble com SIFT (ratio test 0,8 + cross-check), um por limiar do RANSAC. Referência: imagem central (keble_b.jpg); média dos píxeis na sobreposição. No título, o nº de inliers de cada par adjacente. Eixos em píxeis.
