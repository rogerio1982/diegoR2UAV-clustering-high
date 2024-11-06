import numpy as np
from scipy import stats

# Simulações de altitude do UAV (em metros)
simulacoes = [48, 52, 50, 51, 49, 50, 53, 47, 50, 49]

# Número de simulações
n = len(simulacoes)

# Calculando a média
media = np.mean(simulacoes)

# Calculando o desvio padrão (ddof=1 para amostra)
desvio_padrao = np.std(simulacoes, ddof=1)

# Valor Z para 95% de confiança
Z = 1.96

# Intervalo de confiança
erro_padrao = desvio_padrao / np.sqrt(n)
intervalo_confianca = Z * erro_padrao

# Limites do intervalo de confiança
limite_inferior = media - intervalo_confianca
limite_superior = media + intervalo_confianca

print (media, desvio_padrao, limite_inferior, limite_superior)

#Isso significa que, com 95% de confiança, a verdadeira média da altitude do UAV
# estará entre 48,79 metros e 51,01 metros, com base nas 10 simulações realizadas.