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


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


frame_id = (93, 105, 117, 136)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")

for i in range(4):

    mask_4k_loss_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/'

    # render_3d_centerline_path = mask_4k_loss_path + 'frame_' + str(frame_id[i]) + '/render_3d_centerline_frame_' + str(frame_id[i]) + '.npy'
    # render_2d_skeleton_path = mask_4k_loss_path + 'frame_' + str(frame_id[i]) + '/render_2d_skeleton_frame_' + str(frame_id[i]) + '.npy'
    # render_3d_centerline = np.load(render_3d_centerline_path)

    # x_line = render_3d_centerline[:, 0]
    # y_line = render_3d_centerline[:, 1]
    # z_line = render_3d_centerline[:, 2]
    # ax.plot3D(x_line, y_line, z_line, linestyle='-', marker='o', markersize=2.0, linewidth=0.1)

    bezier_surface_vertices_path = mask_4k_loss_path + 'frame_' + str(frame_id[i]) + '/render_3d_bezier_surface_vertices_frame_' + str(frame_id[i]) + '.npy'
    bezier_surface_vertices = np.load(bezier_surface_vertices_path)

    color_list = ('#A7D2CB', '#F2D388', '#C98474', '#874C62')

    ax.scatter(bezier_surface_vertices[:, 0], bezier_surface_vertices[:, 1], bezier_surface_vertices[:, 2], marker='o', s=0.1, color=color_list[i])
    # ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))


frame_id = (93, 94, 95, 97, 98, 104, 105, 108, 109, 110, 111, 112, \
                113, 114, 115, 116, 117, 118, 119, 120, \
                121, 122, 123, 124, 125, 135, 136, 137, \
                138, 139, 140, 147)
traj_pts = []
for i in range(len(frame_id)):

    mask_4k_loss_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/'

    render_3d_centerline_path = mask_4k_loss_path + 'frame_' + str(frame_id[i]) + '/render_3d_centerline_frame_' + str(frame_id[i]) + '.npy'
    # render_2d_skeleton_path = mask_4k_loss_path + 'frame_' + str(frame_id[i]) + '/render_2d_skeleton_frame_' + str(frame_id[i]) + '.npy'
    render_3d_centerline = np.load(render_3d_centerline_path)

    traj_pts.append(render_3d_centerline[-1, :])

traj_pts = np.asarray(traj_pts)
# pdb.set_trace()

ax.plot3D(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2], linestyle='-.', marker='o', markersize=4.0, linewidth=1)
ax.set_title('3d reconstructed curve (mask + 4 keypoints)')

# ax.view_init(0, 23)
ax.view_init(azim=0, elev=23)
axisEqual3D(ax)

ax.grid(True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.tight_layout()
fig.savefig('/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/traj_3d_render_mask_4k.png', dpi=300)

plt.show()