

def calculate_base_station_users_and_throughputs(smalls, users):
    number_of_smalls = len(smalls)
    number_of_users = len(users)

    
    # Cria a matriz que armazenam a id, quantidade de usuários e throughput das base stations
    # As linhas representam as base statios e as colunas as medidas citadas acima, respctivamente
    rows, cols = (number_of_smalls, 3) # dimensões das matrizes
    base_station_users_and_throughputs = [[0 for i in range(cols)] for j in range(rows)]

    for i in range(0,number_of_smalls):
        base_station_throughput = 0 # throughput da base station i
        base_station_total_users = smalls[i].total_users

        # Percorre todos os usuários conectados na base station i
        for j in range(0, base_station_total_users):
            base_station_index = i # índice da base station
            user_id = smalls[i].connected_users[j] # id do usuário conectado
            user_data_rate = users[user_id].data_rate # taxa de dados do usuário
            base_station_throughput = base_station_throughput + user_data_rate # Soma a taxa de dados do usuário conectado ao throughput da base station


        base_station_users_and_throughputs[i][0] = smalls[i].id
        base_station_users_and_throughputs[i][1] = base_station_total_users
        base_station_users_and_throughputs[i][2] = base_station_throughput

    return base_station_users_and_throughputs
