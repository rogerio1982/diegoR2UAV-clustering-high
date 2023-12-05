import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

def PPP_clusters(total_number_of_users, alpha, beta, cov_ue,size):
    
    # Parameters
    num_pico_bs = 5  # Number of pico-BSs
    num_femto_bs = 10  # Number of femto-BSs
    num_sa = 5  # Number of social attractors
    num_ue = total_number_of_users  # Number of UEs
    alpha = alpha #0.5  # Factor for SA movement towards closest BS
    beta = beta #0.5  # Factor for UE movement towards closest SA
    cov_ue = cov_ue #0.5  # Coefficient of variation for UE movement towards closest SA
    
    # Generate pico-BSs and femto-BSs
    bs_x = np.concatenate((
        np.random.uniform(0, size, num_pico_bs),
        np.random.uniform(0, size, num_femto_bs)
    ))
    bs_y = np.concatenate((
        np.random.uniform(0, size, num_pico_bs),
        np.random.uniform(0, size, num_femto_bs)
    ))
    
    # Generate social attractors (SAs)
    sa_x = np.random.uniform(0, size, num_sa)
    sa_y = np.random.uniform(0, size, num_sa)
    
    # Generate UEs
    ue_x = np.random.uniform(0, size, num_ue)
    ue_y = np.random.uniform(0, size, num_ue)
    
    # Calculate distances between SAs and BSs
    sa_bs_dist = np.sqrt((sa_x[:, np.newaxis] - bs_x) ** 2 + (sa_y[:, np.newaxis] - bs_y) ** 2)
    
    # Move SAs towards closest BS
    sa_new_x = alpha * bs_x[np.argmin(sa_bs_dist, axis=1)] + (1 - alpha) * sa_x
    sa_new_y = alpha * bs_y[np.argmin(sa_bs_dist, axis=1)] + (1 - alpha) * sa_y
    
    # Calculate distances between UEs and SAs
    ue_sa_dist = np.sqrt((ue_x[:, np.newaxis] - sa_new_x) ** 2 + (ue_y[:, np.newaxis] - sa_new_y) ** 2)
    
    # Move UEs towards closest SA with added randomness
    ue_new_x = beta * sa_new_x[np.argmin(ue_sa_dist, axis=1)] + (1 - beta) * ue_x + np.random.normal(0, cov_ue, num_ue)
    ue_new_y = beta * sa_new_y[np.argmin(ue_sa_dist, axis=1)] + (1 - beta) * ue_y + np.random.normal(0, cov_ue, num_ue)

    ue_coordinates = np.column_stack((ue_new_x, ue_new_y))
    
#    df1 = pd.DataFrame(ue_new_x)
#    df2 = pd.DataFrame(ue_new_y)
    
#    df_combined = pd.concat([df1, df2], axis=1)
    
    # # Plotting the traffic distribution
    # plt.figure(figsize=(8, 8))
    # #plt.scatter(bs_x, bs_y, marker='o', s=200, c='blue', label='BS')
    # #plt.scatter(sa_new_x, sa_new_y, marker='s', s=100, c='green', label='SA')
    # plt.scatter(ue_new_x, ue_new_y, marker='o', s=10, c='red', label='UE')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Traffic Distribution')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    return ue_coordinates
