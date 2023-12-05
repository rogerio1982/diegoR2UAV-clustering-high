# aqui tem um comparativo, se quiser acrescentar outros que pode fixar sel_n_clusters: https://scikit-learn.org/stable/modules/clustering.html
import os
os.chdir(r"C:\Users\dl_ca\OneDrive - Universidade Federal do Pará - UFPA\Documentos\Python Scripts\UAV-clustering")
import pandas as pd
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
#from genieclust import Genie
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS
#from hdbscan import HDBSCAN
from root import root
from PPP2 import PPP_base
from PPP import PPP_clusters


# Any number can be used in place of '0'.
from IPython import get_ipython
get_ipython().magic('reset -sf')

#data3 = pd.read_csv("data_X.csv")

#Creating Users using k-tier PPP
num_users1 = 100
num_users2 = 150 #50
alpha = 0.5
beta = 0 #0.5
cov_ue = 4
sel_n_clusters = 7
size = 1000

for k in range(0, 5, 1):
    beta = k * 0.25
    
    data1 = PPP_base(num_users1,size)
    data2 = PPP_clusters(num_users2, alpha, beta, cov_ue,size)
    concatenated_ue_coordinates = np.vstack((data1, data2))
    data = pd.DataFrame(concatenated_ue_coordinates)
    data.columns = ['X1','X2']
    
    # List to store the vectors
    vector_list = []
    
    models = [
 #       DBSCAN(eps=50, min_samples=5, metric='euclidean', n_jobs=None, leaf_size=30),
            # erro eh pq ele parece considerar tudo como 1 grupo, teria que ajustar a distancia minima (eps) -> coloquei 50 pra testar
 #       SpectralClustering(n_clusters=sel_n_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=None, n_jobs=None),
            # erro era no root -> mudei affinity='rbf' pra 'nearest_neighbors', parece ter melhorado
        
 #       Genie(n_clusters=sel_n_clusters, gini_threshold=0.3, M=1, exact=True, verbose=0),
        KMeans(n_clusters=sel_n_clusters, random_state=0, verbose=0),
 #       OPTICS(min_samples=5, min_cluster_size=None, metric='minkowski', xi=0.05, n_jobs=None),
 #       HDBSCAN(min_cluster_size=5, min_samples=None, metric='euclidean'),
    ]
    
    size_vetor = len(models)
    
    for model in models:
        try:
            print(model)
            m_name = str(model).split("(")[0]
            
    #         from sklearn.pipeline import make_pipeline
    #         from sklearn.preprocessing import MinMaxScaler
    #         model = make_pipeline(MinMaxScaler(), model)
    
            labels = model.fit_predict(data)
            l_clusters = list(np.unique(labels))
            print(l_clusters)
            if -1 in l_clusters: # o "-1" sao amostras que eles consideram como "outliers"
                l_clusters = l_clusters[1:]
                print(l_clusters)
    
            centr = []
            for c in l_clusters:
                temp_data = data.iloc[labels == c]
                centroid = list(temp_data.mean())
                print("C({}): {} points)".format(c, len(temp_data)))
                centr.append(centroid)
            centr = pd.DataFrame(centr, columns=['X','Y'])
    
            plt.figure()    
            plt.scatter(data["X1"], data["X2"], marker='o', c = 'blue', label='usuario')
            plt.scatter(centr["X"], centr["Y"], marker='o', c = 'red', label='Sc')
            plt.xlabel('Eixo X')
            plt.ylabel('Eixo Y')
            plt.title(m_name +str(beta) + " Clustering")
            plt.legend()
            plt.show()
    
            plt.figure()    
            for c in l_clusters:
                temp_data = data.iloc[labels == c]
                plt.scatter(temp_data["X1"], temp_data["X2"], marker='o', label=c)
            plt.scatter(centr["X"], centr["Y"], marker='o', c = 'k', s=100, label='Sc')
            plt.xlabel('Eixo X')
            plt.ylabel('Eixo Y')
            plt.title(m_name +str(beta) + " Clustering")
        #     plt.legend()
            plt.savefig(m_name +str(beta)+'-Clustering.svg')
            plt.show()        
        except Exception as e:
            print("\nERROR:", e)
    #         raise Exception()
            pass
        
        try:
            uav2 = deepcopy(centr)
            total_number_of_users = len(data);
            ###################################################
            #######AQUI A GENTE VAI ALTERNANDO A ALTURA########
            uav_high = 100
            number_of_small_base_stations = len(uav2);
            results, base_station_users_and_throughputs, total_users_data_rate = root(total_number_of_users, data, number_of_small_base_stations, uav2, uav_high)
            print('\nConexão finalizada do ' + m_name)
            print(". results:", results)
            # Exportar para o excel especificando duas folhas
            tabela = pd.DataFrame(results, columns=[m_name])
#            valores_indice = ['conect', 'desc', 'media', 'MIN', 'MAX', 'MED2', 'MIN2', 'MAX2','UAV', 'conect2']
#            tabela = tabela.set_index(pd.Index(valores_indice))
            vector_list.append(tabela)
            pd.concat(vector_list, axis=1).to_excel(str(beta)+'output.xlsx', index=False)
        except Exception as e:
            print("\nERROR:", e)
    #         raise Exception()
            pass
    
        print("\n{}\n".format("*****"*20))
    #     break
    print("END")