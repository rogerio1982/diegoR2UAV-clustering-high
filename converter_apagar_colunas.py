import pandas as pd

# Caminho do arquivo CSV de entrada e saída
csv_file_path = 'india_todos.csv.csv'
output_csv_path = 'india200.csv'

# Leitura do arquivo CSV
df = pd.read_csv(csv_file_path)

# Remover as colunas 'latitude' e 'longitude'
df = df.drop(columns=['latitude', 'longitude'])

# Salvar o DataFrame resultante em um novo arquivo CSV
df.to_csv(output_csv_path, index=False)

print("Colunas 'latitude' e 'longitude' removidas e arquivo salvo como 'india200.csv'")
