# 3D vector operations and visualisation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define 3D vectors
v1 = np.array([3, 4, 5])
v2 = np.array([2, 1, 6])

# Vector operations
v_add = v1 + v2
v_sub = v1 - v2
dot_product = np.dot(v1, v2)
cross_product = np.cross(v1, v2)

# Set up 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original vectors
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label=f'v1 = {v1}')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label=f'v2 = {v2}')

# Plot cross product
ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], color='g', linestyle='dashed', label='v1 × v2')

# Plot settings
ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_zlim([0, 7])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vectors and Cross Product')
ax.legend()

# Print dot and cross product
print(f"Dot Product (v1 • v2): {dot_product}")
print(f"Cross Product (v1 × v2): {cross_product}")

# Show plot
plt.show()