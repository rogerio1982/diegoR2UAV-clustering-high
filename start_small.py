from base_station import base_station

import math

# Função para calcular o raio do cone
def calcular_raio(h, theta_graus):
    # Convertendo o ângulo de graus para radianos
    theta_radianos = math.radians(theta_graus)

    # Calculando o raio usando a fórmula r = h / tan(theta)
    raio = h / math.tan(theta_radianos)

    return raio


def start_small(number_of_small_base_stations, uav2, uav_high):
    if not isinstance(uav_high, list):
        raise Exception()
    
    # Lista para armazenar as instâncias das small cells base stations
    small_cells = []

    # Contador de small cell
    counter = 0

    #print(uav_high)
    for i in range(0, number_of_small_base_stations):

        new_small_cell = base_station()
        new_small_cell.id = counter # Id do usuário
        new_small_cell.x = uav2.X[i] # Posição no eixo x
        new_small_cell.y = uav2.Y[i] # Posição no eixo y
        new_small_cell.transmit_power = 32 # Potência em dBm
        new_small_cell.frequency = 2.6e9 # Frequência em Ghz
        new_small_cell.base_station_connected = True # Se a base station está ligada ou não
        new_small_cell.total_PRB = 50 # Número total de PRBs
        new_small_cell.remaining_PRB = 50 # Número de PRBs disponíveis
        new_small_cell.bandwidth = 10e6 # Largura de banda em Mhz
#        new_small_cell.coverage_area = calcular_raio(uav_high[i],50)#500 # Área de cobertur r = h / math.tan(theta_radianos)
        new_small_cell.coverage_area = uav_high[i]*math.tan(45)#500 # Área de cobertur r = h / math.tan(theta_radianos)
        #new_small_cell.coverage_area = 500 # Área de cobertur r = h / math.tan(theta_radianos)

        new_small_cell.height = uav_high[i] # Altura da base station
        print ("altura",new_small_cell.height,"raio",new_small_cell.coverage_area)
        #new_small_cell.height = uav_high # Altura da base station

        small_cells.append(new_small_cell) # Adiciona a small cell base station criada na lista de small cells

        counter = counter + 1 # Incrementa a variável contadora

    return small_cells
