

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

import time

# Grava o tempo de início
start_time = time.time()

data = pd.read_csv("india200.csv")

#buscando a melhor solucao de forma incremental.
sel_n_clusters = 6

total=1
resu=[]
distancias = []
fitness=0
#teste = [100,64.7,99.9,55.7,50,58.8,100,50.4,84.5,51.1,93.3,50]
#teste = [30,30,30,30,30,30,30,30,30,30,30,30,30,30]
#teste = [50,50,50,50,50,50,50,50,50,50,50,50,50,50]
#teste = [100,100,100,100,100,100,100,100,100,100,100,100,100,100]

max=200#sel_n_clusters+1
resufinal=[]
tempogravado=[]

random_integer = random.randint(1, 5000)

model = KMeans(n_clusters=sel_n_clusters, random_state=random_integer, verbose=1)

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
    # uav_high = teste
    # uav_high.append(uav_high)

    # calcula o raio de cada cluster
    dis = 0
    for index, user in data.iterrows():
        dist = np.sqrt(np.sum((user - centro) ** 2))
        if dist >= dis:
            dis = dist

    # distancias.append(int(dis*1000))
    calcdist = int((dis / math.tan(45)) * 1000)  # converte raio em altura do uabs
    if calcdist <= 100:
        distancias.append(calcdist)
    else:
        distancias.append(100)
#distancias=[1000,1000]
melhores_alturas=[]
for x in range(1,max):

#altrar as alturas para encontrar otimos locais

    #atribue a distancia encontrada para altura
   # uav_high=distancias#int(raio*1000)#teste  # aqui modifica a altura h=r⋅tan(θ)#altura
    # uav_high.append(int((raio * math.tan(45))*1000))  # aqui modifica a altura h=r⋅tan(θ)
    uav_high = distancias  # int(raio*1000)#teste  # aqui modifica a altura h=r⋅tan(θ)#altura

# uav_high=teste # aqui modifica a altura h=r⋅tan(θ)
    # {results[0]: users max; results[2]: average_user_data_rate max; results[3]: user_minimum_data_rate max}


    #uav_high=30
    #print("dist", distancias) #ajuste das alturas fixas, encontradas pelo algoritmo

    print("dist", distancias)
    centr = pd.DataFrame(centr, columns=['X','Y'])
    uav2 = pd.DataFrame(centr, columns=['X','Y']);
    total_number_of_users = len(data);
    number_of_small_base_stations = len(uav2);
    results, base_station_users_and_throughputs, total_users_data_rate = root(total_number_of_users, data, number_of_small_base_stations, uav2,uav_high)
    print("teste", "resultados",results)
    resu.append(results)
    #resufinal.append(resu)
    total = total + 1

    # Grava o tempo de término
    end_time = time.time()

    # Calcula o tempo decorrido
    elapsed_time = end_time - start_time
    tempogravado.append(elapsed_time)
    print("Tempo decorrido:", elapsed_time, "segundos")


    #total_users_data_rate = calculate_results(data,uav2)
    #count=0
    for x in resu:

        #print(x[2]) #média vazao
        print(x[3])  # vazao minima QoS

        #buscando o melhor troughput da rede levando em consideracao que ja atendeu tdos
        #para isso o algoritmo modifica as posicoes dos drones a cada iteracao
        if x[2] >= fitness:
            fitness = x[2]
            melhores_alturas=uav_high
            distancias = []
            distancias=melhores_alturas
        if x[2] <= fitness:
            melhores_alturas=melhores_alturas
            distancias=[]
            for y in melhores_alturas:
                numero_aleatorio = random.randint(-3, 5)
                value=y-numero_aleatorio
                if value <=50:
                    distancias.append(50)#+numero_aleatorio)
                else:
                    distancias.append(value)#+numero_aleatorio)



resufinal.append(resu)

print("Final")
print(resufinal)
print("melhores fitness: ",fitness)
coluna = [x[2] for x in resu]
print("fitness ordenados")
#for x in coluna:
    #print(x)

print("altura fitness",melhores_alturas)
# Itera sobre as sublistas na lista aninhada
for sublista in resufinal:
    # Itera sobre os elementos em cada sublista e os imprime
    for elemento in sublista:
        print(elemento)


print("Tempo final encontrar solucos:", tempogravado, "segundos")

#users_connection_in_the_small é usado para setar o path loss
#calculate_channel_ATG e calculate_channel.


#######################################################################################

#plot 01
# Número de soluções encontradas
num_solucoes = list(range(1, len(tempogravado) + 1))

# Plotagem do gráfico
plt.figure()
plt.plot(num_solucoes, tempogravado, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Solutions Found')
plt.ylabel('Time (s)')
plt.title('Solutions Found')
plt.grid(True)
#plt.show()


#plot 01
plt.figure()
coluna = [x[0] for x in resu]
indices = list(range(len(resu)))
plt.plot(indices, coluna, marker='o', linestyle='-', color='b', label='Users Connected')
plt.xlabel('UAVBS')
plt.ylabel('Users')
plt.legend()
plt.title('UAVBS connected')
#plt.show()

#plot 02
plt.figure()
coluna = [x[2] for x in resu]
indices = list(range(len(resu)))
plt.plot(indices, coluna, marker='o', linestyle='-', color='b', label='Data Rate')
plt.xlabel('UAVBS')
plt.ylabel('Throughput')
plt.legend()
plt.title('Average Throughput')
#plt.show()

#kmeans 03
plt.figure()
plt.scatter(data["X"], data["Y"], marker='o', c='blue', label='User')
plt.scatter(centr["X"], centr["Y"], marker='x', c='red', label='UAVBS')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.title("Kmeans Clustering")
#plt.show()


# kmeans color 04
plt.figure()
for i in range(len(centr)):
    cluster_data = data[labels == i]
    if i == 0:  # Defina o rótulo apenas na primeira iteração
        plt.scatter(cluster_data["X"], cluster_data["Y"], marker='o', label='Cluster users')
    else:
        plt.scatter(cluster_data["X"], cluster_data["Y"], marker='o')
    #plt.scatter(cluster_data["X"], cluster_data["Y"], marker='o', label='Cluster users')
plt.scatter(centr["X"], centr["Y"], marker='x', c='red', label='UAVBS')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title("Kmeans Clustering ")
#plt.show()

#######################################################################################


'''
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data["X"], data["Y"], 0, marker='o', c='blue', label='User')
ax.scatter(centr["X"], centr["Y"], uav_high, marker='x', c='red', label='UAVBS')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title("3D position UAVBS")
plt.show()
'''


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Criação do gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot para cada cluster com uma cor distinta, similar ao gráfico 2D
for i in range(len(centr)):
    cluster_data = data[labels == i]  # Filtra os dados do cluster atual
    if i==0:
        ax.scatter(cluster_data["X"], cluster_data["Y"], 0, marker='o', label='Cluster users')
    else:
        ax.scatter(cluster_data["X"], cluster_data["Y"], 0, marker='o')


# Plotando os UAV-BS em vermelho e na posição `uav_high`
ax.scatter(centr["X"], centr["Y"], uav_high, marker='x', c='red', label='UAVBS')

# Configurações do gráfico 3D
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title("3D position UAVBS with Cluster Colors")
plt.show()
