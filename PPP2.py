import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

def PPP_base(total_number_of_users,size):

    # Parameters
    num_ue = total_number_of_users  # Number of UEs
    
    # Generate UEs
    ue_x = np.random.uniform(0, size, num_ue)
    ue_y = np.random.uniform(0, size, num_ue)
    
    ue_coordinates = np.column_stack((ue_x, ue_y))

#    df1 = pd.DataFrame(ue_x)
#    df2 = pd.DataFrame(ue_y)
        
#    df_combined = pd.concat([df1, df2], axis=1)
    
    
    # # Plotting the UEs
    # plt.figure(figsize=(8, 8))
    # plt.scatter(ue_x, ue_y, marker='o', s=10, c='red', label='UE')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('UE Distribution')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return ue_coordinates
