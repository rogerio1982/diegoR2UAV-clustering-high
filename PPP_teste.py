import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 100  # Number of points to generate

# Sub-Poisson process (0 < C < 1)
lambda_sub = 5  # Intensity parameter
sub_poisson_points_x = np.random.poisson(lambda_sub, num_points)
sub_poisson_points_y = np.random.poisson(lambda_sub, num_points)

# Poisson process (C = 1)
lambda_poisson = 10  # Intensity parameter
poisson_points_x = np.random.poisson(lambda_poisson, num_points)
poisson_points_y = np.random.poisson(lambda_poisson, num_points)

# Super-Poisson process (C > 1)
lambda_super = 20  # Intensity parameter
super_poisson_points_x = np.random.poisson(lambda_super, num_points)
super_poisson_points_y = np.random.poisson(lambda_super, num_points)

# Plotting space domain realizations
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.scatter(sub_poisson_points_x, sub_poisson_points_y, color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sub-Poisson Process')

plt.subplot(2, 3, 2)
plt.scatter(poisson_points_x, poisson_points_y, color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Poisson Process')

plt.subplot(2, 3, 3)
plt.scatter(super_poisson_points_x, super_poisson_points_y, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Super-Poisson Process')

plt.tight_layout()
plt.show()