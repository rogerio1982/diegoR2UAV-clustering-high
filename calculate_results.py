import numpy as np

def calculate_results(users, smalls):

    number_of_users = len(users)
    number_of_smalls = len(smalls)

    # Cria vetor que armazenam as métricas de sáida
    """
        Índices da variável results:

        1 = Usuários conectados
        2 = Usuários não conectados
        3 = Média taxa de dados dos usuários (Geral)
        4 = Taxa de dados mínima obtidada pelos usuários (Geral)
        5 = Taxa de dados máxima obtidada pelos usuários (Geral)
        6 = Média taxa de dados dos usuários (somente nas small base stations)
        7 = Taxa de dados mínima obtidada pelos usuários (somente nas small base stations)
        8 = Taxa de dados máxima obtidada pelos usuários (somente nas small base stations)
        9 = Small cells ligadas
        10 = Usuários conectados (somente nas small cells base stations)
        
    """
    rows = (10) # tamanho do vetor
    results = [0 for j in range(rows)]

    total_users_data_rate = []
    total_users_data_rate_in_smalls = []

    for i in range(0, number_of_users):

        # Conta a quantidade de usuários conectados e não conectados
        if users[i].user_connected == True:
            results[0] = results[0] + 1
        else:
            results[1] = results[1] + 1

        total_users_data_rate.append(users[i].data_rate)

        if users[i].base_station_type == 1:
            total_users_data_rate_in_smalls.append(users[i].data_rate)

    average_user_data_rate = np.mean(total_users_data_rate)
    user_maximum_data_rate = np.max(total_users_data_rate)
    user_minimum_data_rate = np.min(total_users_data_rate)

    # Verifica se há usuários conectados nas small cells base stations
    if total_users_data_rate_in_smalls:
        average_user_data_rate_in_smalls = np.mean(total_users_data_rate_in_smalls)
        user_maximum_data_rate_in_smalls = np.max(total_users_data_rate_in_smalls)
        user_minimum_data_rate_in_smalls = np.min(total_users_data_rate_in_smalls)
    else:
        average_user_data_rate_in_smalls = 0
        user_maximum_data_rate_in_smalls = 0
        user_minimum_data_rate_in_smalls = 0    

    results[2] = np.round(average_user_data_rate)
    results[3] = np.round(user_minimum_data_rate)
    results[4] = np.round(user_maximum_data_rate)

    results[5] = np.round(average_user_data_rate_in_smalls)
    results[6] = np.round(user_minimum_data_rate_in_smalls)
    results[7] = np.round(user_maximum_data_rate_in_smalls)

    # Conta a quantidade de small cells base station ligadas
    for i in range(0, number_of_smalls):
        if smalls[i].base_station_connected == 1:
            results[8] = results[8] + 1


    for i in range(0, number_of_users):
        if users[i].user_connected == True and users[i].base_station_type == 1:
            results[9] = results[9] + 1


    return results, total_users_data_rate




