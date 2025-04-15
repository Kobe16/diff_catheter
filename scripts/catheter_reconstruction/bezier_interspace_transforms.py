"""
File to perform calculations relating to inverse and 
forward kinematics equations. Calls upon transforms file. 
"""

import numpy as np

def T_matrix(s, delta_x, delta_y, l, r):

    delta = np.sqrt(delta_x**2 + delta_y**2)
    S = np.sin(s * delta / (l * r))
    C = np.cos(s * delta / (l * r))
    
    T11 = 1 + (delta_x**2 / delta**2) * (C - 1)
    T12 = (delta_x * delta_y / delta**2) * (C - 1)
    T13 = -delta_x * S / delta
    T14 = -delta_x * l * r * (1 - C) / delta**2
    
    T21 = (delta_y * delta_x / delta**2) * (C - 1)
    T22 = C + (delta_x**2 / delta**2) * (1 - C)
    T23 = -delta_y * S / delta
    T24 = -delta_y * l * r * (1 - C) / delta**2
    
    T31 = delta_x * S / delta
    T32 = delta_y * S / delta
    T33 = C
    T34 = l * r * S / delta
    
    T41 = 0.0
    T42 = 0.0
    T43 = 0.0
    T44 = 1.0
    
    T = np.array([
        [T11, T12, T13, T14],
        [T21, T22, T23, T24],
        [T31, T32, T33, T34],
        [T41, T42, T43, T44]
    ], dtype=float)
    
    return T

def tendon_disp_to_bezier_control_points(delta_x, delta_y, l, r, p0):
    """
    Convert from constant curvature model to Bezier curve model.
    Computes Bezier control points from given tendon displacement variables.

    Parameters:
    delta_x, delta_y: Tendon displacement variables.
    p0 (array): Base point (homogeneous coordinates, length of 4).

    Returns: First and second Bezier control points (homogeneous coordinates).
    """
    p1 = (2*T_matrix(0.5, delta_x, delta_y, l, r) - T_matrix(0, delta_x, delta_y, l, r) / 2 - T_matrix(1, delta_x, delta_y, l, r) / 2) @ p0
    p2 = T_matrix(1, delta_x, delta_y, l, r) @ p0
    
    return p1, p2



def random_translate_homogeneous_vector(homogeneous_vector, translation_magnitude=0.002):
    # Extract the first three elements (3D coordinates)
    coordinates = homogeneous_vector[:3]
    
    # Generate a random direction as a unit vector
    random_direction = np.random.normal(size=3)
    random_direction /= np.linalg.norm(random_direction)  # Normalize to make it a unit vector
    
    # Create a translation vector with the desired magnitude
    translation_vector = random_direction * translation_magnitude
    
    # Apply the translation to the 3D coordinates
    new_coordinates = coordinates + translation_vector
    
    # Return the new homogeneous vector, with the fourth element still being 1
    new_homogeneous_vector = np.hstack((new_coordinates, 1))
    
    return new_homogeneous_vector

def u_to_b_noisy(delta_x, delta_y, l, r, p0, magnitude=0.002):
    """
    Convert from constant curvature model to Bezier curve model.
    Computes Bezier control points from given tendon displacement variables.

    Parameters:
    delta_x, delta_y: Tendon displacement variables.
    p0 (array): Base point (homogeneous coordinates, length of 4).

    Returns: First and second Bezier control points (homogeneous coordinates).
    """
    c0 = T_matrix(0, delta_x, delta_y, l, r) @ p0
    c1 = T_matrix(0.5, delta_x, delta_y, l, r) @ p0
    c2 = T_matrix(1, delta_x, delta_y, l, r) @ p0
    
    c1 = random_translate_homogeneous_vector(c1, translation_magnitude=magnitude)
    c2 = random_translate_homogeneous_vector(c2, translation_magnitude=magnitude)
    
    p1 = 2 * c1 - 0.5 * c0 - 0.5 * c2
    p2 = c2
    
    return p1, p2

def calculate_radius(A, B, C):
    """
    Calculate the radius of the circumcircle of the triangle formed by points A, B, and C.
    
    Parameters:
    A, B, C: numpy arrays representing the coordinates of the points
    
    Returns:
    radius: The radius of the circumcircle
    circle_center: The center of the circumcircle
    normal_ABC: The normal vector of the plane defined by A, B, and C
    """
    # Calculate vector AB and AC
    vec_AB = B - A
    vec_AC = C - A

    # Normal vector of the plane
    normal_ABC = np.cross(vec_AB, vec_AC)

    # Calculate midpoints of AB and AC
    M_AB = (A + B) / 2
    M_AC = (A + C) / 2

    # Calculate the direction vector of the perpendicular line of AB and AC
    direction_AB = np.cross(normal_ABC, vec_AB)
    direction_AC = np.cross(normal_ABC, vec_AC)
    
    # Calculate the intersection point of the two perpendicular lines (circle center)
    t = (M_AC[0] - M_AB[0]) / (direction_AB[0] + direction_AC[0])
    circle_center = M_AB + t * direction_AB
    
    # Calculate radius
    radius = np.linalg.norm(circle_center - A)
    
    return radius, circle_center, normal_ABC

def angle_between_vectors(A, B):
    """
    Calculate the angle between two vectors A and B.
    
    Parameters:
    A, B: numpy arrays representing the vectors
    
    Returns:
    angle: The angle between vectors A and B in radians
    (Set A as base vector, positive for counterclockwise rotation, negative for clockwise rotation)
    The range of values is -pi to pi.
    """
    # Calculate the dot product of the two vectors
    dot_product = np.dot(A, B)
    # Calculate the norms (magnitudes) of the two vectors
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_A * norm_B)
    # Calculate the absolute value of the angle (in radians)
    angle = np.arccos(cos_theta)
    
    # Calculate the cross product of the two vectors
    cross_product = np.cross(A, B)
    # Determine the rotation direction
    if cross_product[2] < 0:
        # Clockwise rotation, take the negative angle
        angle = -angle
    
    return angle

def bezier_control_points_to_tendon_disp(p_0, p_1, p_2, l, r):
    """
    Convert from Bezier curve model to constant curvature using optimization.
    """
    p_start = p_0[:-1]
    p_mid = p_1[:-1]
    p_end = p_2[:-1]
    
    bezier_mid = 0.5 * (p_mid + 0.5 * p_start + 0.5 * p_end)
    radius, _, n_circle = calculate_radius(bezier_mid, p_start, p_end)
    k = 1 / radius
    delta = k * l * r
    n_base = np.array([0, -1, 0])
    phi = angle_between_vectors(n_base, n_circle)
    
    delta_x_solution = -delta * np.cos(phi)
    delta_y_solution = -delta * np.sin(phi)
        
    return delta_x_solution, delta_y_solution
