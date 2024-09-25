from utils import *

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2'

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
    ax.set_title('Optimization Result')
    ax.legend()
    
result_folder1 = "test_imgs/results_old_07030303"
filename = "bezier_params.npz"
full_path1 = scripts_path + '/' + result_folder1 + '/' + filename

result_folder2 = "test_imgs/results_complete_07030113"
filename = "bezier_params.npz"
full_path2 = scripts_path + '/' + result_folder2 + '/' + filename

# Example of creating two figures
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# Call the function with different data sets
plot_3D_bezier_curve(ax1, full_path1)
plot_3D_bezier_curve(ax2, full_path2)

# Display the figures
plt.show()
    
