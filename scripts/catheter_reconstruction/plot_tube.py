""" Plot 3-D schematic diagram of the Bezier curve in the paper. """

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Generate a tube around the curve
def generate_tube(curve, radius, num_segments=20):
    num_points = len(curve)
    theta = np.linspace(0, 2 * np.pi, num_segments)
    
    # Create a circular cross-section in 3D space (x, y, z)
    circle = np.zeros((num_segments, 3))
    circle[:, 0] = np.cos(theta) * radius  # x-coordinates
    circle[:, 1] = np.sin(theta) * radius  # y-coordinates

    X, Y, Z = [], [], []
    for i in range(num_points):
        if i < num_points - 1:
            tangent = curve[i + 1] - curve[i]
        else:
            tangent = curve[i] - curve[i - 1]
        tangent /= np.linalg.norm(tangent)
        normal = np.cross(tangent, np.array([1, 0, 0]))
        if np.linalg.norm(normal) == 0:
            normal = np.cross(tangent, np.array([0, 1, 0]))
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        rotation_matrix = np.array([normal, binormal, tangent]).T
        
        # Apply rotation and translation to the circular cross-section
        transformed_circle = circle @ rotation_matrix.T + curve[i]
        X.append(transformed_circle[:, 0])
        Y.append(transformed_circle[:, 1])
        Z.append(transformed_circle[:, 2])
    
    return np.array(X), np.array(Y), np.array(Z)


def arc(center, radius, theta_start, theta_end, n_points=100):
    # Generate angle array
    theta = np.linspace(theta_start, theta_end, n_points)

    # Compute arc points in the yz-plane
    x = center[0] + np.zeros_like(theta)  # Keep x-axis constant
    y = center[1] + radius * np.cos(theta)
    z = center[2] + radius * np.sin(theta)

    arc_points = np.vstack((x, y, z)).T  # Reshape to (N, 3)
 
    return arc_points

# Parameters
center = np.array([0, 0, 0])  
R = 5                   
theta_start = 0              
theta_end = 0.55 * np.pi    
n_points = 100              
r = 0.3

arc_points = arc(center, R, theta_start, theta_end, n_points)
uy1 = arc(center, R-r, theta_start, theta_end, n_points)
uy2 = arc(center, R+r, theta_start, theta_end, n_points)
ux1 = arc(center, R, theta_start, theta_end, n_points)
ux1[:, 0] += r
# print(ux1)
ux2 = arc(center, R, theta_start, theta_end, n_points)
ux2[:, 0] -= r

# Generate tube
X_curve, Y_curve, Z_curve = generate_tube(arc_points, r)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

color = '#17becf'

ax.plot_surface(X_curve, Y_curve, Z_curve, color=color, alpha=0.3, rstride=1, cstride=1)
ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='black', linestyle=(0, (3.3, 0.8)),  linewidth=0.3)
ax.plot(uy1[:, 0], uy1[:, 1], uy1[:, 2], color='orange', linewidth=0.3)
ax.plot(uy2[:, 0], uy2[:, 1], uy2[:, 2], color='orange', linewidth=0.3)
ax.plot(ux1[:, 0], ux1[:, 1], ux1[:, 2], color='blue', linewidth=0.3)
ax.plot(ux2[:, 0], ux2[:, 1], ux2[:, 2], color='blue', linewidth=0.3)

ax.axis('equal')

# Set the background to be white
ax.set_axis_off()  # turn off the axis
ax.grid(False) # turn off the grid       

plt.show()