"""
Script that test the camera projection calculation (matching between numerical computation and Blender rendering)
"""

import sys
sys.path.append('E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import numpy as np
import matplotlib.pyplot as plt

from cc_catheter import CCCatheter
from utils import *
import camera_settings

def bezier_curve_2d(P0, P1, P2, num_points=100):
    """
    Compute points on a 2D quadratic Bezier curve
    
    Parameters:
    P0, P1, P2: Control points of the Bezier curve (array-like, 2D coordinates)
    num_points: Number of points to generate on the Bezier curve (default 100)
    
    Returns:
    bezier_points: Numpy array of points on the Bezier curve
    """
    # Define the function to compute a point on the Bezier curve for a given t
    def bezier_curve(t, P0, P1, P2):
        return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
    
    # Generate t values
    t_values = np.linspace(0, 1, num_points)
    
    # Compute all points on the Bezier curve
    bezier_points = np.array([bezier_curve(t, P0, P1, P2) for t in t_values])
    
    return bezier_points

def convert_3d_to_2d(p, camera_extrinsics, fx, fy, cx, cy):
    """
    Convert 3D points to 2D points
    """ 
    p_4d = np.append(p, 1)
    p_cam = camera_extrinsics @ p_4d
    p_x = p_cam[0] * fx / (-p_cam[2]) + cx
    p_y = p_cam[1] * fy / (-p_cam[2]) + cy
    p_2d = np.array([p_x, p_y])
    
    # convert from pixel coordinate in Blender to opencv
    p_2d[0] = round(p_2d[0])
    p_2d[1] = round(480 - p_2d[1])
    
    
    return p_2d

p_0 = np.array([2e-2, 2e-3, 1e-4])
l = 0.2
# r = 0.01
r = 0.01
n_mid_points = 1
n_iter = 50
n_trials = 50
ux = 0.0005
uy = 0.0005 # 0.001

cccatheter = CCCatheter(p_0, l, r, False, False, n_mid_points, n_iter)
cccatheter.set_3dof_params(ux, uy, l)
cccatheter.calculate_cc_points()
cccatheter.calculate_beziers_control_points()

folder_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/blender_imgs/'

file_name = 'test_gt_12'
curve_specs_path = folder_path + file_name + '.npy'
img_save_path = folder_path + file_name + '.png'


# Plot 3D bezier curve
# param_gt = read_gt_params(curve_specs_path)
# p_1 = param_gt[:3]
# p_2 = param_gt[3:]
# control_points = np.vstack([p_0, p_1, p_2])
# plot_3D_bezier_curve(control_points, equal=True)



cccatheter.render_beziers(curve_specs_path, img_save_path, viewpoint_mode=1) # viewpoint_mode
image = plt.imread(img_save_path)
# plt.imshow(image)
# plt.show()

param_gt = read_gt_params(curve_specs_path)
p_1 = param_gt[:3]
p_2 = param_gt[-3:]
control_points = np.vstack([p_0, p_1, p_2])
bezier_3d = bezier_curve_3d(control_points)

bezier_2d = np.zeros((bezier_3d.shape[0], 2))
for i in range(bezier_3d.shape[0]):
    bezier_2d[i] = convert_3d_to_2d(bezier_3d[i], camera_settings.extrinsics, camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y)
    
p2_2d = convert_3d_to_2d(p_2, camera_settings.extrinsics, camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y)
p1_2d = convert_3d_to_2d(p_1, camera_settings.extrinsics, camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y)
p0_2d = convert_3d_to_2d(p_0, camera_settings.extrinsics, camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y)
    
bezier_points = bezier_2d[1:, :]
# bezier_points = bezier_2d

plt.imshow(image)

plt.plot(bezier_points[:, 0], bezier_points[:, 1], color='red', linewidth=2, label='Target Bezier Curve')

plt.scatter([p0_2d[0], p1_2d[0], p2_2d[0]], [p0_2d[1], p1_2d[1], p2_2d[1]], color='blue', label='Control Points')
# Connect control points with dashed lines (P0 to P1, P1 to P2)
plt.plot([p0_2d[0], p1_2d[0]], [p0_2d[1], p1_2d[1]], linestyle='--', color='blue')
plt.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]], linestyle='--', color='blue')

# Set the axis limits to match the image size
plt.xlim(0, 640)  # X axis limit for image width
plt.ylim(480, 0)  # Y axis limit for image height, inverted to match image coordinates

plt.title("Bezier Curve on Image")
plt.axis('off')
plt.legend()
plt.show()


