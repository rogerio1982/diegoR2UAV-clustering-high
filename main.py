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
import math
import matplotlib.pyplot as plt

random.seed(0)


data = pd.read_csv("bairros_filtrados.csv")


#buscando a melhor solucao de forma incremental.
sel_n_clusters = 12

total=1
resu=[]
distancias = []
random_integer = random.randint(1, 5000)
teste = [100,64.7,99.9,55.7,50,58.8,100,50.4,84.5,51.1,93.3,50]

model = KMeans(n_clusters=12, random_state=random_integer, verbose=1)

labels = model.fit_predict(data)
l_clusters = list(np.unique(labels))

centr = []
uav_high = []
centers = model.cluster_centers_

for c in l_clusters:
    temp_data = data.iloc[labels == c]
    centroid = list(temp_data.mean())
    centr.append(centroid)
    centro = centers[c]
    uav_high = teste
    uav_high.append(uav_high)


# calcula o raio de cada cluster
    dis = 0
    for index, user in data.iterrows():
        dist = np.sqrt(np.sum((user - centro) ** 2))
        if dist >= dis:
            dis = dist
distancias.append(dist)
for raio in distancias:
    #uav_high.append(200)  # aqui modifica a altura h=r⋅tan(θ)
    uav_high.append(int((raio * math.tan(45))*1000))  # aqui modifica a altura h=r⋅tan(θ)

   # uav_high=teste # aqui modifica a altura h=r⋅tan(θ)
# {results[0]: users max; results[2]: average_user_data_rate max; results[3]: user_minimum_data_rate max}


#uav_high=30
print("dist", distancias)
centr = pd.DataFrame(centr, columns=['X','Y'])
uav2 = pd.DataFrame(centr, columns=['X','Y']);
total_number_of_users = len(data);
number_of_small_base_stations = len(uav2);
results, base_station_users_and_throughputs, total_users_data_rate = root(total_number_of_users, data, number_of_small_base_stations, uav2,uav_high)
print("teste", "resultados",results)
resu.append(results)
total = total + 1


#total_users_data_rate = calculate_results(data,uav2)
count=0
for x in resu:
    print("estatistica")
    #print(count,x[0],x[2],x[3])
    print(count,x[2],"Mbps")
   # print ("valores")
    #print("qtd=",x[0])
    #print("taxa=", x[2])
   # print("UAVS ON=", x[8])
    count=count+1



'''
users_connection_in_the_small é usado para setar o path loss 
calculate_channel_ATG e calculate_channel.

'''

#######################################################################################
#plot 01
plt.figure()
coluna = [x[0] for x in resu]
indices = list(range(len(resu)))
plt.plot(indices, coluna, marker='o', linestyle='-', color='b', label='Users Connected')
plt.xlabel('UAVBS')
plt.ylabel('Users')
plt.legend()
plt.title('UAVBS connected')
plt.show()

#plot 02
plt.figure()
coluna = [x[2] for x in resu]
indices = list(range(len(resu)))
plt.plot(indices, coluna, marker='o', linestyle='-', color='b', label='Users Connected')
plt.xlabel('UAVBS')
plt.ylabel('Data rate')
plt.legend()
plt.title('Average user data rate')
plt.show()

#kmeans 03
plt.figure()
plt.scatter(data["X"], data["Y"], marker='o', c='blue', label='usuario')
plt.scatter(centr["X"], centr["Y"], marker='x', c='red', label='Sc')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.title("Kmeans Clustering")
plt.show()


# kmeans color 04
plt.figure()
for i in range(len(centr)):
    cluster_data = data[labels == i]
    plt.scatter(cluster_data["X"], cluster_data["Y"], marker='o', label=f'Cluster {i}')
plt.scatter(centr["X"], centr["Y"], marker='x', c='red', label='Centroides')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.title("Kmeans Clustering - Pontos Coloridos por Cluster")
plt.show()

'''
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["X"], data["Y"], 0, marker='o', c='blue', label='usuario')
ax.scatter(centr["X"], centr["Y"], uav_high, marker='x', c='red', label='Sc')
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Eixo Z')
ax.legend()
plt.title("Kmeans Clustering - 3D Visualization")
plt.show()

'''
