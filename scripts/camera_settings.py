""" Define camera settings (intrinsic and extrinsic parameters) for the simulation experiments.
Shoule be consistent with the camera settings in Blender.
"""

import numpy as np

focal_length_x = 10.0
focal_length_y = 10.0

image_size_x = 640.0  ## px
image_size_y = 480.0  ## px
aspect_ratio = image_size_x / image_size_y

sensor_size_x = 7.248  ## (mm) sensor_width in Blender
sensor_size_y = sensor_size_x / aspect_ratio

a = focal_length_x / sensor_size_x * image_size_x
b = focal_length_y / sensor_size_y * image_size_y

center_x = image_size_x / 2.0
center_y = image_size_y / 2.0

intrinsics = np.array([[a, 0, center_x], [0, b, center_y], [0.0, 0.0, 1.0]])

location = np.array([0, 0, 0]).reshape((3, 1)) # camera 1
# location = np.array([-0.3, -0.2, 0]).reshape((3, 1)) # camera 4
rotation_euler = np.array([0, np.pi, np.pi])

# Create rotation matrices for each axis
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(rotation_euler[0]), -np.sin(rotation_euler[0])],
    [0, np.sin(rotation_euler[0]), np.cos(rotation_euler[0])]
])

R_y = np.array([
    [np.cos(rotation_euler[1]), 0, np.sin(rotation_euler[1])],
    [0, 1, 0],
    [-np.sin(rotation_euler[1]), 0, np.cos(rotation_euler[1])]
])

R_z = np.array([
    [np.cos(rotation_euler[2]), -np.sin(rotation_euler[2]), 0],
    [np.sin(rotation_euler[2]), np.cos(rotation_euler[2]), 0],
    [0, 0, 1]
])

# Combine rotations in 'ZYX' order: rotate around X, then Y, then Z
R = R_z @ R_y @ R_x

t = -R.T @ location

# Form the extrinsic matrix
extrinsics = np.hstack((R.T, t))
extrinsics = np.vstack((extrinsics, np.array([[0, 0, 0, 1]])))