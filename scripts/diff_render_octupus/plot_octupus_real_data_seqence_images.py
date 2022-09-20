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

octopus_selected_steps = (0, 1, 40, 70, 100, 130, 160, 200)

fig, axes = plt.subplots(2, 8, figsize=(25, 5))
colormap = mpl.cm.gray
colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
ax = axes.ravel()

CROP_X = 220
CROP_Y = 200

for i in range(len(octopus_selected_steps)):

    if i == 0:
        img_raw_path = '/media/fei/DATA_Fei/Datasets/Octupus_Arm/octupus_data_F/binary_crop/left_recif_raw_136.jpg'
        img_raw_step_converge = cv2.imread(img_raw_path)

        ax[i].imshow(cv2.cvtColor(img_raw_step_converge[0:CROP_X, 0:CROP_Y], cv2.COLOR_BGR2RGB))
        ax[i].set_title('octopus ref')
        ax[i].axis('off')

        continue

    img_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/frame_136/step_converge_' + str(octopus_selected_steps[i]) + '.png'
    img_render_step_converge = cv2.imread(img_path)

    ax[i].imshow(cv2.cvtColor(img_render_step_converge[0:CROP_X, 0:CROP_Y], cv2.COLOR_BGR2RGB))
    title = 'itr=' + str(octopus_selected_steps[i])
    ax[i].set_title(title)
    ax[i].axis('off')

CROP_X = -1
CROP_Y = -1
baxter_selected_steps = (0, 0, 80, 160, 240, 320, 400, 490)
### baxter
for i in range(len(baxter_selected_steps)):

    if i == 0:
        img_raw_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/baxter/rgb.png'
        img_raw_step_converge = cv2.imread(img_raw_path)

        ax[i + len(octopus_selected_steps)].imshow(cv2.cvtColor(img_raw_step_converge[0:CROP_X, 0:CROP_Y], cv2.COLOR_BGR2RGB))
        ax[i + len(octopus_selected_steps)].set_title('baxter ref')
        ax[i + len(octopus_selected_steps)].axis('off')

        continue

    img_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/baxter/' + str(baxter_selected_steps[i]) + '.png'
    img_render_step_converge = cv2.imread(img_path)

    ax[i + len(octopus_selected_steps)].imshow(cv2.cvtColor(img_render_step_converge[0:CROP_X, 0:CROP_Y], cv2.COLOR_BGR2RGB))
    title = 'itr=' + str(baxter_selected_steps[i])
    ax[i + len(octopus_selected_steps)].set_title(title)
    ax[i + len(octopus_selected_steps)].axis('off')

plt.tight_layout()
fig.savefig('/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/render_converge_steps_frame_136.png', dpi=300)

plt.show()