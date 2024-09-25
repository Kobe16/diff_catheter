"""
Test the noise generation method
"""

from bezier_interspace_transforms import *
from utils import *

def generate_noise(reference_point, min_translation_magnitude=0.001, max_translation_magnitude=0.005, noise_scale=1.0, magnitude_variation_factor=10.0):
    """
    Generate a translation vector to account for noise
    Ensure that nearby reference points produce similar translation vectors
    """
    # Scale the reference point to ensure smooth noise generation
    scaled_point = reference_point / noise_scale
    
    # Generate noise based on the coordinates of the scaled point, using sin to generate smooth random values
    noise = np.sin(scaled_point * np.pi * 2)
    
    # Normalize the generated noise to create a unit direction vector
    random_direction = noise / np.linalg.norm(noise)
    
    # Generate a value between 0 and 1 based on smooth noise function
    scaled_magnitude = (np.sin(np.sum(scaled_point * magnitude_variation_factor)) + 1) / 2
    
    # Map the value from [0, 1] to [min_translation_magnitude, max_translation_magnitude]
    translation_magnitude = min_translation_magnitude + (max_translation_magnitude - min_translation_magnitude) * scaled_magnitude
    
    # Multiply the unit direction vector by the random translation magnitude to get the final translation vector
    translation_vector = random_direction * translation_magnitude
    
    print("Noise generated: ", translation_vector, "magnitude: ", np.linalg.norm(translation_vector))
    return translation_vector

# General parameters
p_0 = np.array([2e-2, 2e-3, 0])
p_0_h = np.append(p_0, 1)
r = 0.01
l = 0.2

# Define the ranges for ux and uy
ux_values = np.linspace(0, 0.01, 50)
uy_values = np.linspace(0, 0.01, 50)

# Initialize an array to store the errors
magnitude_mid_matrix = np.zeros((len(ux_values), len(uy_values)))
magnitude_end_matrix = np.zeros((len(ux_values), len(uy_values)))

# Iterate through all combinations of ux and uy
for i, ux in enumerate(ux_values):
    for j, uy in enumerate(uy_values):
        p_1, p_2 = tendon_disp_to_bezier_control_points(ux, uy, l, r, p_0_h)
        vector_mid = generate_noise(p_1[:3], max_translation_magnitude=0.005)
        magnitude_mid = np.linalg.norm(vector_mid)
        vector_end = generate_noise(p_2[:3], max_translation_magnitude=0.005)
        magnitude_end = np.linalg.norm(vector_end)

        magnitude_mid_matrix[i, j] = magnitude_mid
        magnitude_end_matrix[i, j] = magnitude_end


# plt.figure(figsize=(8, 6))
# plt.contourf(ux_values, uy_values, magnitude_mid_matrix.T, levels=50, cmap='viridis')
# plt.colorbar(label='Magnitude')
# plt.xlabel('ux')
# plt.ylabel('uy')
# plt.title('Magnitude of the translation matrix applied to mid point')
# plt.show()


# plt.figure(figsize=(8, 6))
# plt.contourf(ux_values, uy_values, magnitude_end_matrix.T, levels=50, cmap='viridis')
# plt.colorbar(label='Magnitude')
# plt.xlabel('ux')
# plt.ylabel('uy')
# plt.title('Magnitude of the translation matrix applied to end point')
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 6))  

contour1 = ax[0].contourf(ux_values, uy_values, magnitude_mid_matrix.T, levels=50, cmap='viridis')
fig.colorbar(contour1, ax=ax[0], label='Magnitude')
ax[0].set_xlabel('ux')
ax[0].set_ylabel('uy')
ax[0].set_title('Magnitude of the translation matrix applied to mid point')

contour2 = ax[1].contourf(ux_values, uy_values, magnitude_end_matrix.T, levels=50, cmap='viridis')
fig.colorbar(contour2, ax=ax[1], label='Magnitude')
ax[1].set_xlabel('ux')
ax[1].set_ylabel('uy')
ax[1].set_title('Magnitude of the translation matrix applied to end point')

plt.tight_layout()
plt.show()
