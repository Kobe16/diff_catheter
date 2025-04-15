import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from utils import *

def plot_3D_bezier_curve(ax, full_path):
    data = np.load(full_path)
    control_points = data['control_points']
    control_points_gt = data['control_points_gt']
    control_points_init = data['control_points_init']

    # Generate the Bezier curve
    curve = bezier_curve_3d(control_points)
    curve_gt = bezier_curve_3d(control_points_gt)
    curve_init = bezier_curve_3d(control_points_init)
    
    # Plotting the Bezier curve
    ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Optimized Result')

    ax.plot(control_points_gt[:, 0], control_points_gt[:, 1], control_points_gt[:, 2], 'bo--')
    ax.plot(curve_gt[:, 0], curve_gt[:, 1], curve_gt[:, 2], 'b-', label='Ground Truth')

    ax.plot(control_points_init[:, 0], control_points_init[:, 1], control_points_init[:, 2], 'go--')
    ax.plot(curve_init[:, 0], curve_init[:, 1], curve_init[:, 2], 'g-', label='Initial Guess')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title('Optimization Result')
    ax.legend()
    
    return ax

def plot_3d_circle(center, normal, radius, ax):
    # Normalizing the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Generating two orthogonal vectors to the normal
    if (normal == [1, 0, 0]).all():
        v = np.array([0, 1, 0])
    else:
        v = np.cross(normal, [1, 0, 0])
    v = v / np.linalg.norm(v)
    u = np.cross(normal, v)
    
    # Parameter t for the circle
    t = np.linspace(0, 2 * np.pi, 100)
    
    # Circle in 3D space
    circle_points = np.outer(np.cos(t), u * radius) + np.outer(np.sin(t), v * radius) + center
    
    # Plotting
    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], label='Circle in 3D')
    ax.scatter(center[0], center[1], center[2], color='r', label='Center')
    ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], length=radius, color='g', label='Normal vector')
    
    # Setting labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.legend()
    
    return ax


# 示例：给定平面的法向量，圆心坐标和半径
normal_vector = np.array([1, -1, 1])
center = np.array([0.15, -0.25, 0.7])
radius = 0.15

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2'
result_folder = "test_imgs/results_complete_07030113"
filename = "bezier_params.npz"
full_path = scripts_path + '/' + result_folder + '/' + filename

# Example of creating two figures
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Call the function with different data sets
ax = plot_3D_bezier_curve(ax, full_path)

ax = plot_3d_circle(center, normal_vector, radius, ax)

# Display the figures
plt.show()
