import pandas as pd
from pyproj import Proj, transform
import matplotlib.pyplot as plt


def latlon_to_xy(latitude, longitude):
    mercator = Proj(init='epsg:3395')
    x, y = mercator(longitude, latitude)
    return x, y


csv_file_path = 'india.csv'
output_csv_path = 'arquivo_saida.csv'

df = pd.read_csv(csv_file_path)

if 'latitude' in df.columns and 'longitude' in df.columns:
    df['X'], df['Y'] = zip(*df.apply(lambda row: latlon_to_xy(row['latitude'], row['longitude']), axis=1))

    # Salvar o DataFrame resultante em um arquivo CSV
    df.to_csv(output_csv_path, index=False)

    # Plotando o gráfico de dispersão com limites de 0 a 1000
    plt.scatter(df['X'], df['Y'])
    plt.title('Coordenadas Cartesianas')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()
else:
    print("As colunas 'Latitude' e/ou 'Longitude' não foram encontradas no DataFrame.")
