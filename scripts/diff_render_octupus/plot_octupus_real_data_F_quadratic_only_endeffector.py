import sys
from turtle import pd

sys.path.append('..')

import os
import numpy as np

# import transforms
# import bezier_interspace_transforms
from bezier_set import BezierSet
# import camera_settings

import torch
from torch import autograd

import open3d as o3d

import cv2
import matplotlib.pyplot as plt

import pdb

from construction_bezier import ConstructionBezier
from blender_catheter import BlenderRenderCatheter
from diff_render_catheter import DiffRenderCatheter
from loss_define import ContourLoss, MaskLoss, CenterlineLoss
from build_diff_model import DiffOptimizeModel
from dataset_processing import DatasetProcess

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl

from tqdm.notebook import tqdm

from pathlib import Path

dataset_path = '/media/fei/DATA_Fei/Datasets/Octupus_Arm/octupus_data_F/binary_crop/'

mask_4k_final_img = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/'
mask_end_final_img = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_end/'
mask_4k_only_img = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_only/'

fig, axes = plt.subplots(4, 4, figsize=(8, 7))
colormap = mpl.cm.gray
colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
ax = axes.ravel()

frame_id = (93, 105, 117, 136)

for i in range(4):

    img_raw_path = dataset_path + 'left_recif_raw_' + str(frame_id[i]) + '.jpg'
    img_raw_rgb = cv2.imread(img_raw_path)

    ax[i].imshow(cv2.cvtColor(img_raw_rgb, cv2.COLOR_BGR2RGB))
    # ax[i].set_title('Ref raw')
    ax[i].axis('off')

for i in range(4, 8):

    # pdb.set_trace()

    img_path = mask_4k_only_img + 'frame_' + str(frame_id[i - 4]) + '/render_final_frame_' + str(frame_id[i - 4]) + '.png'
    img_render_final_frame = cv2.imread(img_path)

    # pdb.set_trace()

    ax[i].imshow(cv2.cvtColor(img_render_final_frame, cv2.COLOR_BGR2RGB))
    # ax[i].set_title('Ref raw')
    ax[i].axis('off')

for i in range(8, 12):

    # pdb.set_trace()

    img_path = mask_end_final_img + 'frame_' + str(frame_id[i - 8]) + '/render_final_frame_' + str(frame_id[i - 8]) + '.png'
    img_render_final_frame = cv2.imread(img_path)

    # pdb.set_trace()

    ax[i].imshow(cv2.cvtColor(img_render_final_frame, cv2.COLOR_BGR2RGB))
    # ax[i].set_title('Ref raw')
    ax[i].axis('off')

for i in range(12, 16):

    # pdb.set_trace()

    img_path = mask_4k_final_img + 'frame_' + str(frame_id[i - 12]) + '/render_final_frame_' + str(frame_id[i - 12]) + '.png'
    img_render_final_frame = cv2.imread(img_path)

    # pdb.set_trace()

    ax[i].imshow(cv2.cvtColor(img_render_final_frame, cv2.COLOR_BGR2RGB))
    # ax[i].set_title('Ref raw')
    ax[i].axis('off')

plt.tight_layout()
plt.show()
fig.savefig('/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/selected_render_mask_4k.png', dpi=300)
plt.close(fig)
