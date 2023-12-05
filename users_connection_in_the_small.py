from calculate_channel_ATG import calculate_channel_ATG
import numpy as np
from operator import itemgetter
import copy

def users_connection_in_the_small(users, smalls):

    users = copy.deepcopy(users)
    smalls = copy.deepcopy(smalls)

    number_of_smalls = len(smalls) # Número de smalls cells base stations
    number_of_users = len(users) # Número de usuários

    # Cria as matrizes que armazenam a taxa de dados, cqi e sinr dos usuários em relação às antenas
    # As linhas representam os usuários e as colunas as antenas
    rows, cols = (number_of_users, number_of_smalls) # dimensões das matrizes
    data_rate = [[0 for i in range(cols)] for j in range(rows)]
    cqi = [[0 for i in range(cols)] for j in range(rows)]
    sinr = [[0 for i in range(cols)] for j in range(rows)]

    # Calcula o SINR, CQI e DR (1 PRB) de cada usuário para cada small
    for i in range(0, number_of_users):
        for j in range(0, number_of_smalls):
            data_rate[i][j], cqi[i][j], sinr[i][j] = calculate_channel_ATG(users[i], smalls[j], smalls)

    # Dimensão da matriz para armazenar os usuários e a quantidade de PRBs que eles precisam
    # Linhas = quantidade de usuários
    # Colunas 1: id do usuário | Coluna 2: Quantidade de PRB exigida por ele
    rows, cols = (number_of_users, 2) 
    number_of_prbs_requested_by_users = [[0 for i in range(cols)] for j in range(rows)]

    # Identificando a quantidade de PRB requisitadas por cadas usuário
    # A prioridade de conexão será primeiro para os usuários que demandam
    # poucos PRB, ou seja, cujo sinal possui maior qualidade
    for i in range(0, number_of_users):

        # A maior taxa de dados que o usuário i conseguiu atingir utilizando 1 PRB
        # data_rate[i] passa todas as colunas da linha i
        max_user_data_rate = np.max(data_rate[i])
        # Calcula a quantidade de PRBs necessárias para atingir a taxa de dados mínima do usuário i
        if max_user_data_rate != 0:
            user_requested_PRB = np.ceil(users[i].required_data_rate/max_user_data_rate)
        else:
            user_requested_PRB = 0

        # id do usuário
        number_of_prbs_requested_by_users[i][0] = users[i].id
        # quantidade de PRBs necessária
        number_of_prbs_requested_by_users[i][1] = user_requested_PRB

    
    # Organizando em ordem crescente pela quantidade de PRB requisitados de cada usuário
    number_of_prbs_requested_by_users = sorted(number_of_prbs_requested_by_users[0:], key=itemgetter(1))
    
    # Realiza a conexão dos usuários
    for user_id in range(0,number_of_users):
        
        # Variável auxiliar
        helper = 0

        # A conexão inicia pelo usuário que exige a menor quantidade de PRBs
        # i representa o id do usuário
        i = number_of_prbs_requested_by_users[user_id][0]

        # Verifica se o usuário já não está conectado em alguma base station
        if users[i].user_connected == False:

            while helper == 0:
                max_user_data_rate = np.max(data_rate[i]) # A maior taxa de dados que o usuário obteve com 1 PRB
                base_station_index = np.argmax(data_rate[i]) # índice da base com a qual o usuário obteve a maior taxa de dados com 1 PRB
                if max_user_data_rate != 0:
                    user_requested_PRB = np.ceil(users[i].required_data_rate/max_user_data_rate)
                else:
                    user_requested_PRB = 0

                # Se a base station possui PRB suficientes
                if smalls[base_station_index].remaining_PRB >= user_requested_PRB:
                    users[i].data_rate = max_user_data_rate * user_requested_PRB # Taxa de dados final do usuário
                    users[i].used_PRBs = user_requested_PRB # quantidade de PRBs alocadas para o usuário
                    users[i].base_station_id = base_station_index # índice da base station a qual o usuário está conectado
                    users[i].base_station_type = 1 # Tip de base station | Small = 1
                    users[i].CQI = cqi [i][base_station_index] # CQI do usuário
                    users[i].SINR = sinr[i][base_station_index] # SIRN do usuário
                    users[i].user_connected = True # Indica que o usuário está conectado
                    smalls[base_station_index].remaining_PRB = smalls[base_station_index].remaining_PRB - user_requested_PRB # Diminui a quantidade de PRB disponíveis na base station
                    helper = 1 # Altera a variável auxiliar para sair do while
                else:
                    data_rate[i][base_station_index] = 0

                # Caso o usuário não tenha conseguido se conectar com nenhuma small
                if max_user_data_rate == 0:
                    users[i].data_rate = 0 
                    users[i].used_PRBs = 0
                    users[i].base_station_id = 0
                    users[i].CQI = 0
                    users[i].SINR = 0
                    users[i].user_connected = False 
                    helper = 1 # Altera a variável auxiliar para sair do while


    # Conta a quantidade total e quais usuários que estão nas smalls cells base stations
    # A identificação dos usuários é pelo ID
    # j representa as smalls
    # i representa os usuários
    for j in range(0, number_of_smalls):
        counter = 1
        smalls[j].connected_users = []
        for i in range(0,number_of_users):
            if users[i].base_station_id == j and users[i].base_station_type == 1:
                smalls[j].connected_users.append(i) # Adiciona o id do usuário na lista de usuários da base station
                counter = counter + 1

        # Armazena o total de usuários na small j
        smalls[j].total_users = len(smalls[j].connected_users)

                




    return users, smalls