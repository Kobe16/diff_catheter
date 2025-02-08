"""
Plot the workspace of the catheter for a given set of parameters.
"""
import sys
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

from bezier_interspace_transforms import *
from utils import *
import camera_settings_test as camera_settings

def bezier_point(control_points, t):
    """
    Computes a Bezier curve point for a given set of control points and parameter t.
    control_points: numpy array of shape (n, 3), where n is the number of control points in 3D
    t: parameter value (0 <= t <= 1)
    """
    n = len(control_points) - 1
    point = np.zeros(3)
    for i in range(n + 1):
        bernstein_polynomial = (np.math.factorial(n) /
                                (np.math.factorial(i) * np.math.factorial(n - i))) * (t**i) * ((1 - t)**(n - i))
        point += bernstein_polynomial * control_points[i]
    return point

def convert_3d_to_2d(p, camera_extrinsics, fx, fy, cx, cy):
    """
    Convert 3D points to 2D points
    """ 
    p_4d = np.append(p, 1)
    p_cam = camera_extrinsics @ p_4d
    p_x = p_cam[0] * fx / p_cam[2] + cx
    p_y = p_cam[1] * fy / p_cam[2] + cy
    p_2d = np.array([p_x, p_y])
    
    # Convert from blender coordinate system to python matrix coordinates
    p_2d[0] = round(640 - p_2d[0])
    # p_2d[0] = round(p_2d[0])
    # p_2d[1] = round(480 - p_2d[1])
    # p_2d[1] = round(p_2d[1])
    
    return p_2d

"""
Plot 3D workspace of the catheter
"""

# General parameters
p_0 = np.array([2e-2, 2e-3, 0])
p_0_h = np.append(p_0, 1)
r = 0.01
# l = 0.2

# Define the ranges for ux and uy
ux_values = np.linspace(0.00005, 0.003, 30)
uy_values = np.linspace(0.00005, 0.003, 30)


# l_values = np.linspace(0.05, 0.2, 20)
l_values = np.linspace(0.2, 0.2, 1)

# Lists to store the end points and middle points of the Bezier curves
end_points = []
middle_points = []
ux_uy_list = []

# Iterate through all combinations of ux and uy
for ux in ux_values:
    for uy in uy_values:
        for l in l_values:
            # Compute Bezier control points
            p_1, p_2 = tendon_disp_to_bezier_control_points(ux, uy, l, r, p_0_h)
            
            # Convert homogeneous coordinates to 3D points
            p_1_xyz = p_1[:3]
            p_2_xyz = p_2[:3] 
            
            # Collect the end point (p_2) for the point cloud
            end_points.append(p_2_xyz)
            
            # Compute the middle point (t = 0.5) on the Bezier curve
            control_points = np.array([p_0, p_1_xyz, p_2_xyz])
            middle_point = bezier_point(control_points, 0.5)
            middle_points.append(middle_point)
            
            ux_uy_list.append((ux, uy))

# # Convert lists to numpy arrays for easier plotting
# end_points = np.array(end_points)
# middle_points = np.array(middle_points)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the point clouds
# ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], color='blue', label='End points', s=10)
# ax.scatter(middle_points[:, 0], middle_points[:, 1], middle_points[:, 2], color='green', label='Middle points', s=10)

# # Plot p_0 in red
# ax.scatter(p_0[0], p_0[1], p_0[2], color='red', label='p_0 (Start point)', s=50)

# # Set plot labels and titles
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Point Cloud of Catheter Workspace (End Points and Middle Points)')
# ax.legend()

# # Show the plot
# plt.show()


"""
Plot 2D workspace of the catheter
"""
end_points_2d = []

for points in end_points:
    p_2d = convert_3d_to_2d(points, camera_settings.extrinsics, camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y)
    end_points_2d.append(p_2d)
    
# Lists to store ux and uy for points within bounds
in_bounds_ux = []
in_bounds_uy = []

# Check each 2D point to see if it's within the specified bounds
for (ux, uy), (x, y) in zip(ux_uy_list, end_points_2d):
    if 0 <= x <= 640 and 0 <= y <= 480:
        in_bounds_ux.append(ux)
        in_bounds_uy.append(uy)

img_width = 640
img_height = 480 
   
fig, ax = plt.subplots()
    
# 设置图片的边框（这里是640x480）
ax.set_xlim(0, img_width)
ax.set_ylim(0, img_height)

# 将坐标轴调整为左上角原点，x轴向下，y轴向右
ax.invert_yaxis()

# 绘制像素坐标点
for point in end_points_2d:
    ax.plot(point[0], point[1], 'ro')  # 'ro' 表示红色的点
    
# ax.plot(400, 300, 'go')

# 绘制图片的边框
ax.plot([0, img_width, img_width, 0, 0], [0, 0, img_height, img_height, 0], 'b-', lw=2)

# 显示图像
plt.show()


# # Plot the in-bounds ux and uy values
# plt.scatter(in_bounds_ux, in_bounds_uy)
# plt.xlabel('ux')
# plt.ylabel('uy')
# plt.title('In-bounds ux vs uy')
# plt.grid(True)

# # Enable minor ticks
# plt.minorticks_on()
# # Customize major grid
# plt.grid(which='major', linestyle='-', linewidth=0.75, color='black')
# # Customize minor grid
# plt.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')

# plt.show()
        


