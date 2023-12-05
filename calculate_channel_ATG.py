import numpy as np
import math

def calculate_channel_ATG(user, base_station, base_stations):

    number_of_base_stations = len(base_stations)

    # Distância euclidiana
    distance = (((user.x - base_station.x)**2) + ((user.y - base_station.y)**2))**0.5

    #if (distance <= base_station.coverage_area and base_station.base_station_connected):
    if (base_station.base_station_connected):
      
        #ruído branco
        white_noise = 7.4e-13
        # Interefência gerada por outras células
        interference = 0
 
        #reference_distance = 100 # Distância de referência
        #Sv = 9.4 # 8.2 to 10.6 dB
        #speed_of_light = 3e8 # Velocidade da luz (m/s) no vácuo
        #lambda_ = speed_of_light / base_station.frequency # Lambda
        receiving_antenna_height = 1.6 # Altura da antena receptora em metros
        base_station_height = base_station.height # Altura da estação base
        f = base_station.frequency/1e9
        env = 2 
        
        ax=[15,11,5,5]
        bx=[.16,.18,.3,.3]
        a=ax[env]
        b=bx[env]
        #====antenna loss=====================
        A=1 #to calculate with, A=0 to calculate without antenna loss
        #=========max antenna gain=============%
        Go=2.15
        #=============antenna 3db bandwidth=======%
        seta_3db=76  
        #==reflection loss===================%
        L_r=.3
        

        seta = math.atan((base_station_height-base_station_height)/distance)
        
        lost = (-147.5+20*math.log10(f)+20*math.log10(distance)-20*math.log10(math.cos(math.pi/180*seta))) \
        -A*(2*Go-(12*((seta)/seta_3db)**2)-(12*((seta)/seta_3db)**2)) \
    +20*math.log10((10**((-68.8+10*math.log10(f)+10*math.log10(base_station_height-receiving_antenna_height) \
    +20*math.log10(math.cos(math.pi*seta/180))-10*math.log10(1+math.sqrt(2)/(L_r**2 )))/20)*(1-(1/(a*math.exp(-b*(seta-a))+1)))) \
    +(1*(1/(a*math.exp(-b*(seta-a))+1))))

        received_power = 10**((base_station.transmit_power - lost)/10)/1000

        # Calcula a interferência intercelula
        for i in range(0, number_of_base_stations):
            if (base_stations[i].base_station_connected and base_stations[i].id != base_station.id):
                distance_a = (((base_stations[i].x - user.x)**2) + ((base_stations[i].y - user.y)**2))**0.5
                lost_a = base_stations[i].transmit_power - (92.45 + 20 * math.log10(distance_a/1000) + 20 * math.log10(f/1e9))
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
         


