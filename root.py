from start_scenario import start_scenario
from users_connection_in_the_small import users_connection_in_the_small
from calculate_base_station_users_and_throughputs import calculate_base_station_users_and_throughputs
from calculate_results import calculate_results

"""
    Função que retorna a saída com os resultados obtidos da modelagem. Os resultados são armazenados em uma matriz (24x10)
    As linhas representam as horas.
    As colunas representam o seguinte:
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

def root(total_number_of_users, data, number_of_small_base_stations, uav2, uav_high):

    # Criação do cenário
    users, smalls = start_scenario(total_number_of_users, data, number_of_small_base_stations, uav2, uav_high)

    # Matriz que armazena os resultados para os 24 períodos
    rows, cols = (1, 10) # dimensões das matrizes
    results = [[0 for i in range(cols)] for j in range(rows)]

    # Conexão dos usuários na small base station
    users_after_connection_in_small, smalls_after_connection = users_connection_in_the_small(users, smalls)
    # Calcula a quantidade e taxa de usuários em cada small base station
    base_station_users_and_throughputs = calculate_base_station_users_and_throughputs(smalls_after_connection, users_after_connection_in_small)

    # Calcula os resultados
    results, total_users_data_rate = calculate_results(users_after_connection_in_small, smalls_after_connection)

    return results, base_station_users_and_throughputs, total_users_data_rate 