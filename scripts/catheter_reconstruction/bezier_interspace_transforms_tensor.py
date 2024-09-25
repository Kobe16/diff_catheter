"""
File to perform calculations relating to inverse and 
forward kinematics equations. Calls upon transforms file. 
"""

import torch

# def T_matrix(s, delta_x, delta_y, l, r, gpu_or_cpu):
#     delta = torch.sqrt(delta_x**2 + delta_y**2)
#     S = torch.sin(s * delta / (l * r))
#     C = torch.cos(s * delta / (l * r))
    
#     T11 = 1 + (delta_x**2 / delta**2) * (C - 1)
#     T12 = (delta_x * delta_y / delta**2) * (C - 1)
#     T13 = -delta_x * S / delta
#     T14 = delta_x * l * r * (1 - C) / delta**2
    
#     T21 = (delta_y * delta_x / delta**2) * (C - 1)
#     T22 = 1 + (delta_y**2 / delta**2) * (C - 1)
#     T23 = -delta_y * S / delta
#     T24 = delta_y * l * r * (1 - C) / delta**2
    
#     T31 = delta_x * S / delta
#     T32 = delta_y * S / delta
#     T33 = C
#     T34 = l * r * S / delta
    
#     T41 = torch.tensor(0.0, dtype=torch.float32).to(gpu_or_cpu)
#     T42 = torch.tensor(0.0, dtype=torch.float32).to(gpu_or_cpu)
#     T43 = torch.tensor(0.0, dtype=torch.float32).to(gpu_or_cpu)
#     T44 = torch.tensor(1.0, dtype=torch.float32).to(gpu_or_cpu)
    
#     T = torch.stack([
#         torch.stack([T11, T12, T13, T14]),
#         torch.stack([T21, T22, T23, T24]),
#         torch.stack([T31, T32, T33, T34]),
#         torch.stack([T41, T42, T43, T44])
#     ])
    
#     return T

def T_matrix(s, delta_x, delta_y, l, r, gpu_or_cpu):
    delta = torch.sqrt(delta_x**2 + delta_y**2)
    S = torch.sin(s * delta / (l * r))
    C = torch.cos(s * delta / (l * r))
    
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
    
    T41 = torch.tensor(0.0, dtype=torch.float).to(gpu_or_cpu)
    T42 = torch.tensor(0.0, dtype=torch.float).to(gpu_or_cpu)
    T43 = torch.tensor(0.0, dtype=torch.float).to(gpu_or_cpu)
    T44 = torch.tensor(1.0, dtype=torch.float).to(gpu_or_cpu)
    
    T = torch.stack([
        torch.stack([T11, T12, T13, T14]),
        torch.stack([T21, T22, T23, T24]),
        torch.stack([T31, T32, T33, T34]),
        torch.stack([T41, T42, T43, T44])
    ])
    
    return T

def tendon_disp_to_bezier_control_points(delta_x, delta_y, l, r, p0, gpu_or_cpu):
    # print("delta_x dtype:", delta_x.dtype)
    # print("delta_y dtype:", delta_y.dtype)
    # print("l dtype:", l.dtype)
    # print("r dtype:", r.dtype)
    # print("p0 dtype:", p0.dtype)
    # print("T_matrix(0.5, delta_x, delta_y, l, r, gpu_or_cpu) dtype:", T_matrix(0.5, delta_x, delta_y, l, r, gpu_or_cpu).dtype)
    p1 = (2*T_matrix(0.5, delta_x, delta_y, l, r, gpu_or_cpu) - T_matrix(0, delta_x, delta_y, l, r, gpu_or_cpu) / 2 - T_matrix(1, delta_x, delta_y, l, r, gpu_or_cpu) / 2) @ p0
    p2 = T_matrix(1, delta_x, delta_y, l, r, gpu_or_cpu) @ p0
    
    return p1, p2

def calculate_radius(A, B, C):
    vec_AB = B - A
    vec_AC = C - A

    normal_ABC = torch.linalg.cross(vec_AB, vec_AC)

    M_AB = (A + B) / 2
    M_AC = (A + C) / 2

    direction_AB = torch.linalg.cross(normal_ABC, vec_AB)
    direction_AC = torch.linalg.cross(normal_ABC, vec_AC)
    
    t = (M_AC[0] - M_AB[0]) / (direction_AB[0] + direction_AC[0])
    circle_center = M_AB + t * direction_AB
    
    radius = torch.norm(circle_center - A)
    
    return radius, circle_center, normal_ABC

def angle_between_vectors(A, B):
    dot_product = torch.dot(A, B)
    norm_A = torch.norm(A)
    norm_B = torch.norm(B)
    cos_theta = dot_product / (norm_A * norm_B)
    angle = torch.acos(cos_theta)
    
    cross_product = torch.linalg.cross(A, B)
    if cross_product[2] < 0:
        angle = -angle
    
    return angle

def bezier_control_points_to_tendon_disp(p_0, p_1, p_2, l, r, gpu_or_cpu):
    p_start = p_0[:-1]
    p_mid = p_1[:-1] # bezier middle control point
    p_end = p_2[:-1]
    
    bezier_mid = 0.5 * (p_mid + 0.5 * p_start + 0.5 * p_end) # bezier middle point
    radius, _, n_circle = calculate_radius(bezier_mid, p_start, p_end)
    k = 1 / radius
    delta = k * l * r
    n_base = torch.tensor([0, -1, 0], dtype=torch.float32).to(gpu_or_cpu)
    phi = angle_between_vectors(n_base, n_circle)
    
    delta_x_solution = -delta * torch.cos(phi)
    delta_y_solution = -delta * torch.sin(phi)
        
    return delta_x_solution, delta_y_solution
