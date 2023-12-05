import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os

# Any number can be used in place of '0'.
from IPython import get_ipython
get_ipython().magic('reset -sf')


# Pasta contendo os arquivos Excel
pasta = "C:/Users/dl_ca/OneDrive - Universidade Federal do Pará - UFPA/Documentos/Python Scripts/UAV-clustering"

# Lista para armazenar os dataframes de cada arquivo
dataframes = []

# Iterar sobre os arquivos na pasta
for arquivo in os.listdir(pasta):
    if arquivo.endswith(".xlsx"):
        # Ler o arquivo Excel
        caminho_arquivo = os.path.join(pasta, arquivo)
        df = pd.read_excel(caminho_arquivo)
        
        # Adicionar o dataframe à lista
        dataframes.append(df)

# Combinar os dataframes em um único dataframe
df_final = pd.concat(dataframes)

# Exibir o dataframe final
print(df_final)



# # Dados dos algoritmos
# beta = [5430307, 2070608, 2356628, 2593093, 2737704]
# dbscan = [2411059, 2345984, 2412216, 2568898, 2611441]
# spectral = [2383169, 2374935, 2372111, 2501761, 2480500]
# genie = [2403283, 2398632, 2414145, 2567570, 2617719]
# kmeans = [2347175, 2314250, 2332118, 2347024, 2727926]
# optics = [2042291, 2358722, 2269937, 2531731, 2705046]
# hdbscan = [0, 0.25, 0.5, 0.75, 1]

# # Valores do parâmetro
# param_values = [0, 0.25, 0.5, 0.75, 1]

# # Definindo a largura das barras
# bar_width = 0.1

# # Definindo a posição das barras no eixo x
# bar_positions = np.arange(len(param_values))

# # Plotando as barras para cada algoritmo
# plt.bar(bar_positions - 3*bar_width, beta, width=bar_width, label='Beta')
# plt.bar(bar_positions - 2*bar_width, dbscan, width=bar_width, label='DBSCAN')
# plt.bar(bar_positions - bar_width, spectral, width=bar_width, label='SpectralClustering')
# plt.bar(bar_positions, genie, width=bar_width, label='Genie')
# plt.bar(bar_positions + bar_width, kmeans, width=bar_width, label='KMeans')
# plt.bar(bar_positions + 2*bar_width, optics, width=bar_width, label='OPTICS')
# plt.bar(bar_positions + 3*bar_width, hdbscan, width=bar_width, label='HDBSCAN')

# # Definindo os rótulos dos eixos
# plt.xlabel('Valor do parâmetro')
# plt.ylabel('Resultados')

# # Definindo os rótulos do eixo x
# plt.xticks(bar_positions, param_values)

# # Adicionando uma legenda
# plt.legend()

# # Exibindo o gráfico
# plt.show()
