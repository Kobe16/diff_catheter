import sys
sys.path.append('..')
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from test_loss_define_v2 import GenerateRefData
from test_reconst_v2 import ConstructionBezier
from utils import *

""" Specify the ground truth image """

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2'
dataset_folder = "gt_dataset4"
gt_name = 'gt_90_0.0002_-0.0002_0.2_0.01'

case_naming = scripts_path + '/' + dataset_folder + '/' + gt_name
img_save_path = case_naming + '.png'
cc_specs_path = case_naming + '.npy'

"""Test image processing"""

img_ref_binary = process_image(img_save_path)
plt.figure(1)
plt.imshow(img_ref_binary)
# plt.show()

"""Test reference centerline and contour"""
image_ref = torch.from_numpy(img_ref_binary.astype(np.float32))
generate_ref_data = GenerateRefData(image_ref)

# Get the reference centerline
ref_catheter_centerline = generate_ref_data.get_raw_centerline()

img_raw_skeleton = ref_catheter_centerline.flip(1)
ref_skeleton_tip_point = img_raw_skeleton[0, :] # tip
ref_skeleton_base_point = img_raw_skeleton[-1, :] # base
tip_coords = ref_skeleton_tip_point.numpy().astype(int)
base_coords = ref_skeleton_base_point.numpy().astype(int)

x_centerline = img_raw_skeleton.numpy()[:, 0]
y_centerline = img_raw_skeleton.numpy()[:, 1]

# # downsample the centerline
# x_centerline = img_raw_skeleton.numpy()[:, 0][::10]
# y_centerline = img_raw_skeleton.numpy()[:, 1][::10]

# Get the reference contour
ref_catheter_contour = generate_ref_data.get_raw_contour()
x_contour = ref_catheter_contour.numpy()[:, 0]
y_contour = ref_catheter_contour.numpy()[:, 1]

plt.figure(2)
plt.imshow(img_ref_binary)
plt.scatter(tip_coords[0], tip_coords[1], color='green', label='Tip')  
plt.scatter(base_coords[0], base_coords[1], color='red', label='Base')
# plt.plot(x_centerline, y_centerline, marker='o', linestyle='-', color='b', label='Reference Centerline', markersize=0.5)
plt.scatter(x_centerline, y_centerline, c='b', s=3, label='Reference Centerline')
plt.scatter(x_contour, y_contour, c='orange', s=3, label='Reference Contour')
plt.title('Reference Centerline and Contour')
plt.legend()
# plt.show()

"""Test projected centerline and contour"""

if torch.cuda.is_available():
    gpu_or_cpu = torch.device("cuda:0") 
    torch.cuda.set_device(gpu_or_cpu)
else:
    gpu_or_cpu = torch.device("cpu")
    
p_start_np = np.load(cc_specs_path)[0, 0, :] + 1e-8
p_start = torch.tensor(p_start_np).to(gpu_or_cpu)
para_gt_np = read_gt_params(cc_specs_path)
para_init = nn.Parameter(torch.from_numpy(para_gt_np).to(gpu_or_cpu),
                                    requires_grad=True)

save_img_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/test_imgs/rendered_imgs_old/initial_frame_11.png'

build_bezier = ConstructionBezier()
build_bezier.to(gpu_or_cpu)
build_bezier.loadRawImage(img_save_path)
build_bezier.getBezierCurveCylinder(p_start, para_init)
build_bezier.getCylinderMeshProjImg()
build_bezier.getBezierProjImg()
build_bezier.draw2DCylinderImage(image_ref, save_img_path)

bezier_proj_img = build_bezier.bezier_proj_img.detach().numpy()
bezier_proj_centerline_img = build_bezier.bezier_proj_centerline_img.detach().numpy()
img_render_point_cloud = bezier_proj_img.reshape(bezier_proj_img.shape[0] * bezier_proj_img.shape[1], 2)

projected_base = (int(bezier_proj_centerline_img[1, 0]), int(bezier_proj_centerline_img[1, 1]))
projected_tip = (int(bezier_proj_centerline_img[-1, 0]), int(bezier_proj_centerline_img[-1, 1]))
x_proj_centerline = bezier_proj_centerline_img[1:, 0]
y_proj_centerline = bezier_proj_centerline_img[1:, 1]

mask = (img_render_point_cloud[:, 0] >= 0) & (img_render_point_cloud[:, 0] <= 640) & (img_render_point_cloud[:, 1] >= 0) & (img_render_point_cloud[:, 1] <= 480)
img_render_point_cloud = img_render_point_cloud[mask]
x_proj_contour = img_render_point_cloud[1:, 0]
y_proj_contour = img_render_point_cloud[1:, 1]

plt.figure(3)
plt.imshow(img_ref_binary)

plt.scatter(projected_tip[0], projected_tip[1], color='green', label='Projected Tip')  
plt.scatter(projected_base[0], projected_base[1], color='red', label='Projected Base')

plt.plot(x_proj_centerline, y_proj_centerline, marker='o', linestyle='-', color='b', label='Projected Centerline', markersize=3)
plt.plot(x_proj_contour, y_proj_contour, marker='o', linestyle='-', color='orange', label='Projected Contour', markersize=2)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Projected Centerline and Contour')
plt.legend()
plt.grid(True)

""" Draw bezier courves in 3D"""
para_gt_np_4 = np.load(cc_specs_path)
para_gt_np_4 = np.squeeze(para_gt_np_4)
para_gt_np_3 = bezier_conversion_4_to_3(para_gt_np_4)
print("Ground truth parameters: ", para_gt_np_3)

def random_deviation(point, min_distance=0.01, max_distance=0.05):
    # Generate a random direction as a unit vector
    random_direction = np.random.randn(3)
    random_direction /= np.linalg.norm(random_direction)
    # Generate a random length between min_distance and max_distance
    random_length = np.random.uniform(min_distance, max_distance)
    # Calculate the offset
    offset = random_direction * random_length
    # Return the point after applying the offset
    return point + offset, random_length

# para_init_np = None
para_init_np = para_gt_np_3.copy()
para_init_np[1], deviation1 = random_deviation(para_gt_np_3[1])
para_init_np[2], deviation2 = random_deviation(para_gt_np_3[2])

# para_init_np = np.array([[0.02, 0.002, 0],
#                    [0.02, 0.001, 0.406],
#                    [-0.19, 0.003, 0.85]], dtype=np.float32)

if para_init_np is not None:
    print("Initialized parameters: ", para_init_np)
    print("Deviation of middle control point: ", deviation1)
    print("Deviation of end point: ", deviation2)
    plot_3D_bezier_curve(control_points_gt=para_gt_np_3, control_points_init=para_init_np)
else:
    plot_3D_bezier_curve(control_points_gt=para_gt_np_3)
    
plt.show()

