# Legendas — BRIEF próprio

Geradas automaticamente com os números da corrida.

**brief_pattern_GI.png**

Padrão de testes da estratégia G I do BRIEF próprio: X e Y uniformes no patch, Uniforme(−S/2, S/2). Cada segmento liga os dois píxeis comparados num teste (bit = 1 se p(x) < p(y)); 256 testes, patch de 48×48 px, seed 0. Comprimento médio dos pares: 24,3 px; testes distintos: 256; testes degenerados (x = y): 0. Coordenadas relativas ao centro do patch, em píxeis.

**brief_pattern_GII.png**

Padrão de testes da estratégia G II do BRIEF próprio: X e Y gaussianos centrados no centro do patch, variância S²/25. Cada segmento liga os dois píxeis comparados num teste (bit = 1 se p(x) < p(y)); 256 testes, patch de 48×48 px, seed 0. Comprimento médio dos pares: 15,9 px; testes distintos: 255; testes degenerados (x = y): 0. Coordenadas relativas ao centro do patch, em píxeis.

**brief_pattern_GIII.png**

Padrão de testes da estratégia G III do BRIEF próprio: X gaussiano (variância S²/25) e Y gaussiano centrado em X (variância S²/100). Cada segmento liga os dois píxeis comparados num teste (bit = 1 se p(x) < p(y)); 256 testes, patch de 48×48 px, seed 0. Comprimento médio dos pares: 5,5 px; testes distintos: 256; testes degenerados (x = y): 3. Coordenadas relativas ao centro do patch, em píxeis.

**brief_pattern_GV.png**

Padrão de testes da estratégia G V do BRIEF próprio: X fixo no centro e Y em todos os pontos de uma grelha polar (8 anéis × 32 ângulos). Cada segmento liga os dois píxeis comparados num teste (bit = 1 se p(x) < p(y)); 256 testes, patch de 48×48 px, seed 0. Comprimento médio dos pares: 13,5 px; testes distintos: 240; testes degenerados (x = y): 0. Coordenadas relativas ao centro do patch, em píxeis.
