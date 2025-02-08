""" Plot 3D Bezier curve (with surface). Control experiment. """

import sys
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D

from utils import *
import camera_settings
import path_settings

exp_name = 'EXP003'
i = 6

data_alias = 'D' + str(0).zfill(2)
method_dir = os.path.join(path_settings.results_dir, exp_name)
data_dir = os.path.join(method_dir, data_alias + '_' + str(i).zfill(4))
print(data_dir)

params_save_path = data_dir + '/' + 'bezier_params.pkl'
print(params_save_path)

def plot_3D_bezier_curve(ax, full_path, radius=0.002, num_segments=20):
    with open(params_save_path, 'rb') as file:
        data = pickle.load(file)
    control_points_opt = data[-2]
    control_points_gt = data[-1]
    control_points_init = data[0]
    control_points_1 = data[1]
    control_points_2 = data[2]

    # Generate the Bezier curve
    curve_opt = bezier_curve_3d(control_points_opt)
    curve_gt = bezier_curve_3d(control_points_gt)
    curve_init = bezier_curve_3d(control_points_init)
    curve_1 = bezier_curve_3d(control_points_1)
    curve_2 = bezier_curve_3d(control_points_2)

    # Generate a tube around the curve
    def generate_tube(curve, radius, num_segments):
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
        
        return [np.array(X), np.array(Y), np.array(Z)]
    
    # Generate tube
    tube_opt = generate_tube(curve_opt, radius, num_segments)
    tube_init = generate_tube(curve_init, radius, num_segments)
    tube_1 = generate_tube(curve_1, radius, num_segments)
    tube_2 = generate_tube(curve_2, radius, num_segments)
    

    # Plotting the Bezier curve
    # ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--')
    # ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Optimized Result')
    
    color_gt = '#bcbd22'
    color_init = '#E0F7FA'  
    color_1 = '#ADD8E6'
    color_2 = '#4682B4'
    color_opt = '#4682B4' 
    fontsize = 14

    ax.plot(control_points_gt[:, 0], control_points_gt[:, 1], control_points_gt[:, 2], marker='o', linestyle='--', color=color_gt)
    ax.plot(curve_gt[:, 0], curve_gt[:, 1], curve_gt[:, 2], color=color_gt, linestyle='-', label='Ground Truth')

    # ax.plot(control_points_init[:, 0], control_points_init[:, 1], control_points_init[:, 2], 'go--')
    # ax.plot(curve_init[:, 0], curve_init[:, 1], curve_init[:, 2], 'g-', label='Initial Guess')

    # Plotting the tube for the optimized curve
    ax.plot_surface(tube_opt[0], tube_opt[1], tube_opt[2], color=color_opt, alpha=1, rstride=1, cstride=1)
    ax.plot_surface(tube_init[0], tube_init[1], tube_init[2], color=color_init, alpha=0.5, rstride=1, cstride=1)
    ax.plot_surface(tube_1[0], tube_1[1], tube_1[2], color=color_1, alpha=0.5, rstride=1, cstride=1)
    ax.plot_surface(tube_2[0], tube_2[1], tube_2[2], color=color_2, alpha=0.5, rstride=1, cstride=1)
    
    # Create fake legend
    legend_line1 = Line2D([0], [0], color=color_gt, linestyle='-', label='Target Curve')
    legend_line2 = Line2D([0], [0], color=color_init, linestyle='-', label='Initial Guess')
    legend_line3 = Line2D([0], [0], color=color_opt, linestyle='-', label='Optimized Result')
    ax.legend(handles=[legend_line1, legend_line2, legend_line3], bbox_to_anchor=(0.59, 0.87), fontsize=fontsize+2) # bbox_to_anchor=(0.59, 0.97)

    # Set labels
    # ax.set_title('Optimization Result')
    # ax.set_aspect('equal')
    
    ax.set_facecolor((1, 1, 1))
    ax.xaxis.pane.fill = False   # 关闭 x 轴背景
    ax.yaxis.pane.fill = False   # 关闭 y 轴背景
    ax.zaxis.pane.fill = False   # 关闭 z 轴背景

    # 设置 x, y, z 轴的刻度间隔
    ax.set_xticks(np.arange(-0.16, 0.16, 0.05))  # 设置 x 轴刻度为 0 到 10，间隔 2
    ax.set_yticks(np.arange(-0.1, 0.2, 0.05))  # 设置 y 轴刻度为 0 到 20，间隔 5
    ax.set_zticks(np.arange(0, 1, 0.1))  # 设置 z 轴刻度为 0 到 15，间隔 3
    
    ax.tick_params(axis='z', pad=10)  # 调整 z 轴刻度标签的距离，单位为像素, 20
    # 设置轴标注内容和位置
    ax.set_xlabel("x (m)", labelpad=10, fontsize=fontsize)  # 设置 X 轴标注，增加距离, 15
    ax.set_ylabel("y (m)", labelpad=10, fontsize=fontsize)  # 设置 Y 轴标注，增加距离, 15
    ax.set_zlabel("z (m)", labelpad=15, fontsize=fontsize)  # 设置 Z 轴标注，增加距离, 45
    
    ax.tick_params(axis="both", labelsize=fontsize)

    # 设置网格间隔
    ax.grid(True, linestyle='--', linewidth=0.5)  # 启用网格并设置样式
    
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')

# Call the function with different data sets
plot_3D_bezier_curve(ax1, params_save_path)
# plot_3D_bezier_curve(ax2, full_path2)

# Display the figures
plt.show()