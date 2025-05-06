# TODO: ao apresentar apenas tabelas de output codes com a máxima menor distância, filtrar também para apresentar só as com maior média de distância de Hamming entre seus pares

import numpy as np
from scipy.spatial.distance import pdist, squareform

numclasses = 2
numbits = 5
lim_min_dist = 0

"""
classe, bits, dmin, quant
2, 2, 2, 1
2, 3, 3, 1
2, 4, 4, 1
2, 5, 5, 1

3, 3, 2, 3
3, 4, 2, 39
3, 5, 3, 45
3, 6, 4, 45
3, 7, 4, 840
3, 8, 5, 840
3, 9, 6, 840
3, 10, 6, 19425
3, 11, 7, 17325

4, 4, 2, 57
4, 5, 3, 15
4, 6, 4, 15
4, 7, 4, 2800
4, 8, 5, 280
4, 9, 6, 280
4, 10, 6, 143675
"""

def gen_ecoc_matriz(seed):
    sseed = seed
    filtro = (2 ** numbits) - 1
    lastval = seed & filtro
    for j in range(numclasses-2):
        sseed = (sseed >> numbits) 
        val = sseed & filtro
        if val <= lastval:
            return None
        lastval = val

    bits = np.array(list(f"{seed:032b}"), dtype=int)
    ecoc_matrix = ([np.ones(numbits)])
    ini = 32 - (numclasses-1) * numbits
    for j in range(numclasses-1):
        ecoc_matrix.append(bits[ini:ini + numbits])
        ini += numbits
    return np.array(ecoc_matrix).astype(int)

solucoes = []
quant = (2**numbits) ** (numclasses-1)
for i in range(quant):
#for i in range(128*128-1,128*128+10):
    ecoc_matrix = gen_ecoc_matriz(i)
    if ecoc_matrix is None:
        continue
    # Calcula a matriz de distâncias de Hamming
    distances = pdist(ecoc_matrix, metric='hamming') * ecoc_matrix.shape[1]
    min_dist = int(np.min(distances))
    if(min_dist < lim_min_dist):
        #print(f"{i} {min_dist}")
        #print(i, ecoc_matrix)
        continue

    if(min_dist > lim_min_dist):
        lim_min_dist = min_dist
        solucoes = []
    
    solucoes.append(i)

#for i in solucoes:
    print(f"seed = {i} {gen_ecoc_matriz(i)}")

print(f"{numclasses}, {numbits}, {lim_min_dist}, {len(solucoes)}")
        
