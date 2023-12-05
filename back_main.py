import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from genieclust import Genie
from sklearn.cluster import KMeans
from root import root
#from clustering_algoritms import clustering_algoritm
import random



random.seed(0)




data = pd.read_csv("data_X.csv")

sel_n_clusters = 6

#Aqui vou inserir as clusterizacoes
#########################
model = Genie(n_clusters=sel_n_clusters, gini_threshold=0.3, M=1, exact=True, verbose=1)

labels = model.fit_predict(data)
l_clusters = list(np.unique(labels))

centr = []
uav_high=[]
for c in l_clusters:
    temp_data = data.iloc[labels == c]
    centroid = list(temp_data.mean())
    centr.append(centroid)
    uav_high.append(30)
centr = pd.DataFrame(centr, columns=['X','Y'])

plt.scatter(data["X1"], data["X2"], marker='o', c = 'blue', label='usuario')
plt.scatter(centr["X"], centr["Y"], marker='o', c = 'red', label='Sc')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.title("Genie Clustering")
plt.show()


uav2 = pd.DataFrame(centr, columns=['X','Y']);
total_number_of_users = len(data);
number_of_small_base_stations = len(uav2);
results, base_station_users_and_throughputs, total_users_data_rate = root(total_number_of_users, data, number_of_small_base_stations, uav2,uav_high)
print('result',results)
###########################