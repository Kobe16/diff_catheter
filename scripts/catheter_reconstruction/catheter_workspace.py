"""
Plots the workspace of the catheter for a given set of parameters.
"""

from bezier_interspace_transforms import *
from utils import *

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

# General parameters
p_0 = np.array([2e-2, 2e-3, 0])
p_0_h = np.append(p_0, 1)
r = 0.01
l = 0.2

# Define the ranges for ux and uy
ux_values = np.linspace(0.0005, 0.005, 50)
uy_values = np.linspace(0.0005, 0.005, 50)

# uy_values = np.linspace(0.0002, 0.02, 50)

# Lists to store the end points and middle points of the Bezier curves
end_points = []
middle_points = []

# Iterate through all combinations of ux and uy
for ux in ux_values:
    for uy in uy_values:
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

# Convert lists to numpy arrays for easier plotting
end_points = np.array(end_points)
middle_points = np.array(middle_points)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point clouds
ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], color='blue', label='End points', s=10)
ax.scatter(middle_points[:, 0], middle_points[:, 1], middle_points[:, 2], color='green', label='Middle points', s=10)

# Plot p_0 in red
ax.scatter(p_0[0], p_0[1], p_0[2], color='red', label='p_0 (Start point)', s=50)

# Set plot labels and titles
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud of Catheter Workspace (End Points and Middle Points)')
ax.legend()

# Show the plot
plt.show()
