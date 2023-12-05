import numpy as np

def calculate_channel(user, base_station, base_stations):

    number_of_base_stations = len(base_stations)

    # Distância euclidiana
    distance = (((user.x - base_station.x)**2) + ((user.y - base_station.y)**2))**0.5

    if (distance <= base_station.coverage_area and base_station.base_station_connected):
        
        #ruído branco
        white_noise = 7.4e-13
        # Interefência gerada por outras células
        interference = 0
 
        reference_distance = 100 # Distância de referência
        Sv = 9.4 # 8.2 to 10.6 dB
        speed_of_light = 3e8 # Velocidade da luz (m/s) no vácuo
        lambda_ = speed_of_light / base_station.frequency # Lambda
        receiving_antenna_height = 1.6 # Altura da antena receptora em metros
        base_station_height = base_station.height # Altura da estação base
        equalizer = 16 # Equalizador

        # Parâmetros do cenário SUI tipo C
        a = 3.6
        b = 0.005
        c = 20

        A = 20 * np.log10(4 * np.pi * reference_distance/lambda_)
        Y = a - b * base_station_height + ( c / base_station_height)

        lost = A + 10 * Y * np.log10(distance/reference_distance) + Sv - equalizer

        received_power = 10**((base_station.transmit_power - lost)/10)/1000

        # Calcula a interferência intercelula
        for i in range(0, number_of_base_stations):
            if (base_stations[i].base_station_connected and base_stations[i].id != base_station.id):
                distance_a = (((base_stations[i].x - user.x)**2) + ((base_stations[i].y - user.y)**2))**0.5
                lost_a = base_stations[i].transmit_power - (A + 10 * Y * np.log10(distance_a/reference_distance) + Sv - equalizer)
                interference = interference + (10**(lost_a/10)) / 1000

        sinr = (received_power / (white_noise + interference))

        C = base_station.bandwidth / base_station.total_PRB
        data_rate = (C * np.log2(1 + sinr))
        cqi = round(1 + ((7/13)*(sinr+6)))


    else:
        sinr = 0
        data_rate = 0
        cqi = 0
        interference = 0


    
    return data_rate, cqi, sinr
         


