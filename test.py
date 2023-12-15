import pandas as pd

# Carregar o arquivo CSV
caminho_arquivo_entrada = 'arquivo_saida.csv'  # Substitua pelo caminho real do seu arquivo CSV
caminho_arquivo_saida = 'novo_arquivo.csv'  # Substitua pelo caminho desejado para o novo arquivo CSV

# Carregar o CSV para um DataFrame
dados = pd.read_csv(caminho_arquivo_entrada)

# Excluir as duas primeiras colunas
dados_sem_primeiras_colunas = dados.iloc[:, 2:]

# Salvar o DataFrame resultante em um novo arquivo CSV
dados_sem_primeiras_colunas.to_csv(caminho_arquivo_saida, index=False)
