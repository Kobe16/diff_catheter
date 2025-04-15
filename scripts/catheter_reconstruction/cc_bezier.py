"""
Visualize the conversion of constant curvature curve to bezier curve
"""

from bezier_interspace_transforms import *
from utils import *

p_0 = np.array([2e-2, 2e-3, 0])
p_0_h = np.append(p_0, 1)
r = 0.01
ux = 0.005
uy = -0.001 
l = 0.2

cc_curve = []
for s in np.linspace(0, 1, 100):    
    cc_point = T_matrix(s, ux, uy, l, r) @ p_0_h
    cc_curve.append(cc_point[:3])
cc_curve = np.array(cc_curve)


p_1, p_2 = tendon_disp_to_bezier_control_points(ux, uy, l, r, p_0_h)
bezier_control_points = np.vstack([p_0, p_1[:3], p_2[:3]])
bezier_curve = bezier_curve_3d(bezier_control_points)

p_1_noisy, p_2_noisy = u_to_b_noisy(ux, uy, l, r, p_0_h, magnitude=0.01)
bezier_control_points_noisy = np.vstack([p_0, p_1_noisy[:3], p_2_noisy[:3]])
bezier_curve_noisy = bezier_curve_3d(bezier_control_points_noisy)


# Calculate the Euclidean distance for each pair of corresponding points
distances = np.linalg.norm(cc_curve - bezier_curve, axis=1)
# Calculate the mean distance
mean_error = np.mean(distances)
print("Average error:", mean_error)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(bezier_control_points[:, 0], bezier_control_points[:, 1], bezier_control_points[:, 2], 'ro--')
ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], bezier_curve[:, 2], 'r-', label='Bezier Curve')

ax.plot(bezier_control_points_noisy[:, 0], bezier_control_points_noisy[:, 1], bezier_control_points_noisy[:, 2], 'go--')
ax.plot(bezier_curve_noisy[:, 0], bezier_curve_noisy[:, 1], bezier_curve_noisy[:, 2], 'g-', label='Bezier Curve (Noisy)')

ax.plot(cc_curve[:, 0], cc_curve[:, 1], cc_curve[:, 2], 'b-', label='Constant Curvature Curve')
ax.plot(cc_curve[len(cc_curve) // 2, 0], cc_curve[len(cc_curve) // 2, 1], cc_curve[len(cc_curve) // 2, 2], 'bo')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Bezier and Non-Bezier Curves')
ax.set_aspect('equal')
ax.legend()

plt.show()


"""
The error of the approximation of bezier curve with constant curvature curve with different values of ux and uy.
"""

# # Define the ranges for ux and uy
# ux_values = np.linspace(-0.01, 0.01, 50)
# uy_values = np.linspace(-0.01, 0.01, 50)

# # Initialize an array to store the errors
# error_matrix = np.zeros((len(ux_values), len(uy_values)))

# # Iterate through all combinations of ux and uy
# for i, ux in enumerate(ux_values):
#     for j, uy in enumerate(uy_values):
#         # Recalculate cc_curve and bezier_curve for each ux, uy pair
#         cc_curve = []
#         for s in np.linspace(0, 1, 100):
#             cc_point = T_matrix(s, ux, uy, l, r) @ p_0_h
#             cc_curve.append(cc_point[:3])
#         cc_curve = np.array(cc_curve)

#         p_1, p_2 = tendon_disp_to_bezier_control_points(ux, uy, l, r, p_0_h)
#         bezier_control_points = np.vstack([p_0, p_1[:3], p_2[:3]])
#         bezier_curve = bezier_curve_3d(bezier_control_points)

#         # Calculate the error between the two curves
#         distances = np.linalg.norm(cc_curve - bezier_curve, axis=1)
#         mean_error = np.mean(distances)

#         # Store the mean error in the matrix
#         error_matrix[i, j] = mean_error

# # Create a heatmap of the error
# plt.figure(figsize=(8, 6))
# plt.contourf(ux_values, uy_values, error_matrix.T, levels=50, cmap='viridis')
# plt.colorbar(label='Mean Error')
# plt.xlabel('ux')
# plt.ylabel('uy')
# plt.title('Error between cc_curve and bezier_curve')
# plt.show()
    

