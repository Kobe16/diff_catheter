""" Plot 3D Bezier curve (with surface). Shape reconstruction. """

from utils import *
from matplotlib.lines import Line2D

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/gt_dataset6/figs'

# data = np.load(full_path)

# control_points = data['control_points']
# control_points_gt = data['control_points_gt']
# control_points_init = data['control_points_init']

# plot_3D_bezier_curve(control_points, control_points_gt, control_points_init)

# def plot(path):
#     data = np.load(path)

#     control_points = data['control_points']
#     control_points_gt = data['control_points_gt']
#     control_points_init = data['control_points_init']

#     plot_3D_bezier_curve(control_points, control_points_gt, control_points_init)

# result_folder = "test_imgs/results_old_07030303"
# filename = "bezier_params.npz"
# full_path = scripts_path + '/' + result_folder + '/' + filename
# plot(full_path)

# result_folder = "test_imgs/results_complete_07030113"
# filename = "bezier_params.npz"
# full_path = scripts_path + '/' + result_folder + '/' + filename
# plot(full_path)



# def plot_3D_bezier_curve(ax, full_path):
#     data = np.load(full_path)
#     control_points = data['control_points']
#     control_points_gt = data['control_points_gt']
#     control_points_init = data['control_points_init']

#     # Generate the Bezier curve
#     curve = bezier_curve_3d(control_points)
#     curve_gt = bezier_curve_3d(control_points_gt)
#     curve_init = bezier_curve_3d(control_points_init)
    
#     # Plotting the Bezier curve
#     ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--')
#     ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Optimized Result')

#     ax.plot(control_points_gt[:, 0], control_points_gt[:, 1], control_points_gt[:, 2], 'bo--')
#     ax.plot(curve_gt[:, 0], curve_gt[:, 1], curve_gt[:, 2], 'b-', label='Ground Truth')

#     ax.plot(control_points_init[:, 0], control_points_init[:, 1], control_points_init[:, 2], 'go--')
#     ax.plot(curve_init[:, 0], curve_init[:, 1], curve_init[:, 2], 'g-', label='Initial Guess')

#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Optimization Result')
#     ax.legend()

def plot_3D_bezier_curve(ax, full_path, radius=0.002, num_segments=20):
    data = np.load(full_path)
    control_points = data['control_points']
    control_points_gt = data['control_points_gt']
    control_points_init = data['control_points_init']

    # Generate the Bezier curve
    curve = bezier_curve_3d(control_points)
    curve_gt = bezier_curve_3d(control_points_gt)
    curve_init = bezier_curve_3d(control_points_init)

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
        
        return np.array(X), np.array(Y), np.array(Z)
    
    # Generate tube
    X_curve, Y_curve, Z_curve = generate_tube(curve, radius, num_segments)
    X_curve_init, Y_curve_init, Z_curve_init = generate_tube(curve_init, radius, num_segments)

    # Plotting the Bezier curve
    # ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--')
    # ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Optimized Result')
    
    color_opt = '#4682B4' # '#17becf'
    color_gt = '#bcbd22'
    color_init = '#ADD8E6' #'#7f7f7f'

    ax.plot(control_points_gt[:, 0], control_points_gt[:, 1], control_points_gt[:, 2], marker='o', linestyle='--', color=color_gt)
    ax.plot(curve_gt[:, 0], curve_gt[:, 1], curve_gt[:, 2], color=color_gt, linestyle='-', label='Ground Truth')

    # ax.plot(control_points_init[:, 0], control_points_init[:, 1], control_points_init[:, 2], 'go--')
    # ax.plot(curve_init[:, 0], curve_init[:, 1], curve_init[:, 2], 'g-', label='Initial Guess')

    # Plotting the tube for the optimized curve
    ax.plot_surface(X_curve, Y_curve, Z_curve, color=color_opt, alpha=1, rstride=1, cstride=1)
    ax.plot_surface(X_curve_init, Y_curve_init, Z_curve_init, color=color_init, alpha=0.5, rstride=1, cstride=1)
    
    # Create fake legend
    legend_line1 = Line2D([0], [0], color=color_gt, linestyle='-', label='Target Curve')
    legend_line2 = Line2D([0], [0], color=color_init, linestyle='-', label='Initial Guess')
    legend_line3 = Line2D([0], [0], color=color_opt, linestyle='-', label='Optimized Result')
    ax.legend(handles=[legend_line1, legend_line2, legend_line3], bbox_to_anchor=(0.59, 0.97))

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimization Result')
    # ax.legend()
    # ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('equal')
    
    ax.set_facecolor((1, 1, 1))
    ax.xaxis.pane.fill = False   # Disable x-axis background
    ax.yaxis.pane.fill = False   # Disable y-axis background
    ax.zaxis.pane.fill = False   # Disable z-axis background

    # Set tick intervals for x, y, z axes
    ax.set_xticks(np.arange(-0.16, 0, 0.05))  # X-axis ticks
    ax.set_yticks(np.arange(-0.16, 0, 0.05))  # Y-axis ticks
    ax.set_zticks(np.arange(0, 1, 0.1))  # Z-axis ticks

    ax.tick_params(axis='z', pad=20)  # Adjust z-axis tick label spacing
       
    # Set axis labels with padding
    ax.set_xlabel("x (m)", labelpad=15)  
    ax.set_ylabel("y (m)", labelpad=15)  
    ax.set_zlabel("z (m)", labelpad=45)  

    ax.grid(True, linestyle='--', linewidth=0.5)  # Set grid style
    
    

    
result_folder1 = "results_01211907/iter_0"
filename = "bezier_params.npz"
full_path1 = scripts_path + '/' + result_folder1 + '/' + filename

# result_folder2 = "test_imgs/results_complete_07030113"
# filename = "bezier_params.npz"
# full_path2 = scripts_path + '/' + result_folder2 + '/' + filename

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')

# Call the function with different data sets
plot_3D_bezier_curve(ax1, full_path1)
# plot_3D_bezier_curve(ax2, full_path2)

# Display the figures
plt.show()
    
