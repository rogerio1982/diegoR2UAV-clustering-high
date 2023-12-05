import os #teatee
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from genieclust import Genie
from sklearn.cluster import KMeans
#from hdbscan import HDBSCAN
from root import root
from calculate_results import calculate_results
#from clustering_algoritms import clustering_algoritm
import random


random.seed(0)


data = pd.read_csv("bairros_filtrados.csv")


#buscando a melhor solucao de forma incremental.
sel_n_clusters = 3

total=1
resu=[]
for x in range(sel_n_clusters):
    model = KMeans(n_clusters=total, random_state=0, verbose=1)

    labels = model.fit_predict(data)
    l_clusters = list(np.unique(labels))

    centr = []
    uav_high = []
    for c in l_clusters:
        temp_data = data.iloc[labels == c]
        centroid = list(temp_data.mean())
        centr.append(centroid)
        uav_high.append(30)#aqui modifica a altura
    centr = pd.DataFrame(centr, columns=['X','Y'])

    uav2 = pd.DataFrame(centr, columns=['X','Y']);
    total_number_of_users = len(data);
    number_of_small_base_stations = len(uav2);
    results, base_station_users_and_throughputs, total_users_data_rate = root(total_number_of_users, data, number_of_small_base_stations, uav2,uav_high)
    print("teste",x, "resultados",results)
    resu.append(results)
    total = total + 1

#total_users_data_rate = calculate_results(data,uav2)
count=0
for x in resu:
    print(count,x)
   # print ("valores")
    #print("qtd=",x[0])
    #print("taxa=", x[2])
   # print("UAVS ON=", x[8])
    count=count+1

plt.scatter(data["X1"], data["X2"], marker='o', c='blue', label='usuario')
plt.scatter(centr["X"], centr["Y"], marker='x', c='red', label='Sc')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.title("Kmeans Clustering")
plt.show()

###########################

