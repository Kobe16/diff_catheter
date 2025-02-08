""" Test the camera projection in the catheter shape reconstruction pipeline. """

import sys
sys.path.append('..')
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from collections import deque
import pickle
import os

from scripts.catheter_reconstruction.construction_bezier import ConstructionBezier
from catheter_reconstruction.loss_define import (
    ContourChamferLoss, 
    TipDistanceLoss, 
    GenerateRefData
)
from catheter_reconstruction.catheter_motion_tensor import CatheterMotion
from catheter_reconstruction.utils import *


if torch.cuda.is_available():
    gpu_or_cpu = torch.device("cuda:0") 
    torch.cuda.set_device(gpu_or_cpu)
else:
    gpu_or_cpu = torch.device("cpu")

# case_naming = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/gt_dataset5/gt_16_0.0006_0.0002_0.2_0.01'
case_naming = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/blender_imgs/test_gt_13'
img_save_path = case_naming + '.png'
cc_specs_path = case_naming + '.npy'

para_gt_np = read_gt_params(cc_specs_path)
print(para_gt_np)

p_start = torch.tensor([2e-2, 2e-3, 1e-8]).to(gpu_or_cpu) # 0 here will cause NaN in draw2DCylinderImage, pTip
# para_init = np.array([0.13, 0.12, 0.9, 0.06, 0.033, 0.5],
#                     dtype=np.float32)
para_init = nn.Parameter(torch.from_numpy(para_gt_np).to(gpu_or_cpu),
                                      requires_grad=True)


img_ref_binary = process_image(img_save_path)

image_ref = torch.from_numpy(img_ref_binary.astype(np.float32)).to(gpu_or_cpu)
print(image_ref.shape)

# save_img_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/test_imgs/rendered_imgs_old/initial_frame_10.png'
save_img_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/blender_imgs/initial_frame.png'

build_bezier = ConstructionBezier()
build_bezier.to(gpu_or_cpu)
build_bezier.loadRawImage(img_save_path)
build_bezier.getBezierCurveCylinder(p_start, para_init)
build_bezier.getCylinderMeshProjImg()
build_bezier.getBezierProjImg()
build_bezier.draw2DCylinderImage(image_ref, save_img_path)

bezier_proj_img = build_bezier.bezier_proj_img.detach().numpy()
bezier_proj_centerline_img = build_bezier.bezier_proj_centerline_img.detach().numpy()
print("Shape of bezier_proj_img", bezier_proj_img.shape)
print("Shape of bezier_proj_centerline_img", bezier_proj_centerline_img.shape)
img_render_point_cloud = bezier_proj_img.reshape(bezier_proj_img.shape[0] * bezier_proj_img.shape[1], 2)
print("Shape of img_render_point_cloud", img_render_point_cloud.shape)

projected_base = (int(bezier_proj_centerline_img[1, 0]), int(bezier_proj_centerline_img[1, 1]))
projected_tip = (int(bezier_proj_centerline_img[-1, 0]), int(bezier_proj_centerline_img[-1, 1]))
print("Projected tip", projected_tip)

x = bezier_proj_centerline_img[1:, 0]
y = bezier_proj_centerline_img[1:, 1]

# x_contour = img_render_point_cloud[1:, 0]
# y_contour = img_render_point_cloud[1:, 1]

# plt.figure(figsize=(8, 6))
plt.imshow(img_ref_binary)

plt.scatter(projected_tip[0], projected_tip[1], color='yellow', label='Projected Tip')  
plt.scatter(projected_base[0], projected_base[1], color='red', label='Projected Base')
plt.scatter(400, 300, color='green', label='Reference Point')

plt.plot(x, y, marker='o', linestyle='-', color='b', label='Projected Centerline', markersize=3)
# plt.scatter(x_contour, y_contour, label='Projected Contour')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Catheter Centerline Plot')
plt.legend()
plt.grid(True)
plt.show()

image = plt.imread(save_img_path)
plt.imshow(image)
plt.show()