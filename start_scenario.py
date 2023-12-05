from start_user import start_user
from start_small import start_small
import matplotlib.pyplot as plt
import numpy as np


def start_scenario(total_number_of_users, data, number_of_small_base_stations, uav2, uav_high):
    
    X_users = np.linspace(0,0,total_number_of_users)
    Y_users = np.linspace(0,0,total_number_of_users)
    X_small = np.linspace(0,0,number_of_small_base_stations)
    Y_small = np.linspace(0,0,number_of_small_base_stations)
    
    # Cria os usuários
    users = start_user(total_number_of_users, data)
    
    # Cria as small cells base stations
    small_cells = start_small(number_of_small_base_stations, uav2, uav_high)
    
    for i in range (0,total_number_of_users):
        X_users[i] = users[i].x 
        Y_users[i] = users[i].y 
    
    for i in range (0,number_of_small_base_stations):
        X_small[i] = small_cells[i].x 
        Y_small[i] = small_cells[i].y 
        
    # plt.scatter(X_users, Y_users, marker='o', c = 'blue', label='usuario')
    # plt.scatter(X_small, Y_small, marker='o', c = 'red', label='Sc')
    # plt.xlabel('Eixo X')
    # plt.ylabel('Eixo Y')
    # plt.title('Mapa')
    # plt.legend()
    # plt.show()

    return users, small_cells
