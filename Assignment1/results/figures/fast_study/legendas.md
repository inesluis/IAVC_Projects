# Legendas — estudo do FAST

Geradas automaticamente com os números da corrida.

**fast_repeatability_i.png**

Estudo do detetor FAST em iluminação (HPatches i_*): repetibilidade em função do threshold de intensidade (eixo logarítmico), com NMS ligada e desligada; linha horizontal: SIFT (valores por omissão). Sem limite de pontos. Média sobre os 188 pares (de 285) em que todas as configurações foram avaliadas: a repetibilidade não é calculada quando uma imagem tem mais de 50 000 pontos, e só se comparam configurações nos mesmos pares. Repetibilidade = correspondências ground-truth 1-para-1 a menos de 3 px / mínimo dos pontos na zona comum. Cinzento: nível ao acaso, a mesma medida com a homografia ground-truth deslocada (+20, +20) px e (-20, -20) px (média).

**fast_N_kps_mean_i.png**

Estudo do detetor FAST em iluminação (HPatches i_*): nº de keypoints por imagem em função do threshold de intensidade (eixo logarítmico), com NMS ligada e desligada; linha horizontal: SIFT (valores por omissão). Sem limite de pontos. Média sobre 285 pares.

**fast_t_det_ms_i.png**

Estudo do detetor FAST em iluminação (HPatches i_*): tempo de deteção por imagem em função do threshold de intensidade (eixo logarítmico), com NMS ligada e desligada; linha horizontal: SIFT (valores por omissão). Sem limite de pontos. Média sobre 285 pares.

**fast_repeatability_v.png**

Estudo do detetor FAST em ponto de vista (HPatches v_*): repetibilidade em função do threshold de intensidade (eixo logarítmico), com NMS ligada e desligada; linha horizontal: SIFT (valores por omissão). Sem limite de pontos. Média sobre os 113 pares (de 295) em que todas as configurações foram avaliadas: a repetibilidade não é calculada quando uma imagem tem mais de 50 000 pontos, e só se comparam configurações nos mesmos pares. Repetibilidade = correspondências ground-truth 1-para-1 a menos de 3 px / mínimo dos pontos na zona comum. Cinzento: nível ao acaso, a mesma medida com a homografia ground-truth deslocada (+20, +20) px e (-20, -20) px (média).

**fast_N_kps_mean_v.png**

Estudo do detetor FAST em ponto de vista (HPatches v_*): nº de keypoints por imagem em função do threshold de intensidade (eixo logarítmico), com NMS ligada e desligada; linha horizontal: SIFT (valores por omissão). Sem limite de pontos. Média sobre 295 pares.

**fast_t_det_ms_v.png**

Estudo do detetor FAST em ponto de vista (HPatches v_*): tempo de deteção por imagem em função do threshold de intensidade (eixo logarítmico), com NMS ligada e desligada; linha horizontal: SIFT (valores por omissão). Sem limite de pontos. Média sobre 295 pares.
