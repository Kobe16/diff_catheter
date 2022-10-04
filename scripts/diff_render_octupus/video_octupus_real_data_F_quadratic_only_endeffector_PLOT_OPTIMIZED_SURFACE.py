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


if __name__ == '__main__':

    for i in range(0, 32):

        selected_frame_id = i

        sim_case = '/media/fei/DATA_Fei/Datasets/Octupus_Arm/octupus_data_F/binary_crop/'
        frame_id_list = (93, 94, 95, 97, 98, 104, 105, 108, 109, 110, 111, 112, \
                        113, 114, 115, 116, 117, 118, 119, 120, \
                        121, 122, 123, 124, 125, 135, 136, 137, \
                        138, 139, 140, 147)
        ## 0-93  / 1-94  / 2-95  / 3-97  / 4-98/
        ## 5-104 / 6-105 / 7-108 / 8-109 / 9-110/
        ## 10-111 / 11-112 / 12-113 / 13-114 / 14-115
        ## 15-116 / 16-117 / 17-118 / 18-119 / 19-120
        ## 20-121 / 21-122 / 22-123 / 23-124 / 24-125
        ## 25-135 / 26-136 / 27-137 / 28-138 / 29-139
        ## 30-140 / 31-147

        frame_id = frame_id_list[selected_frame_id]

        optimized_para_path = '/home/fei/icra2023_diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/version_paper/frame_' + str(frame_id)

        frame_naming = sim_case + 'left_recif_binary_' + str(frame_id)
        img_save_path = frame_naming + '.jpg'
        img_raw_path = sim_case + 'left_recif_raw_' + str(frame_id) + '.jpg'

        cylinder_primitive_path = '/home/fei/icra2023_diff_catheter/scripts/diff_render_octupus/shape_primitive/cylinder_primitve_101_40.obj'

        ###========================================================
        ### Optimization Rendering
        ###========================================================
        ## Set the cuda device
        if torch.cuda.is_available():
            gpu_or_cpu = torch.device("cuda:0")
            torch.cuda.set_device(gpu_or_cpu)
        else:
            gpu_or_cpu = torch.device("cpu")

        # gpu_or_cpu = torch.device("cpu")

        radius_gt = np.load(sim_case + 'radius_data.npy')
        radius_gt_3d = torch.from_numpy(radius_gt).to(gpu_or_cpu)

        cam_RT_H_mat = np.load(sim_case + 'left_cam_RT_H_crop.npy')
        cam_K_mat = np.load(sim_case + 'left_cam_K_crop.npy')

        ### loading ground truth data
        # gt_centline_3d = np.load(sim_case + 'centerlines/centerline' + str(frame_id) + '.npy')
        gt_centline_3d = np.load(sim_case + 'centerlines_denosie/centerline_denosie_' + str(frame_id) + '.npy')
        # gt_centline_3d_if_flip = np.flip(gt_centline_3d, 0).copy()
        gt_centline_3d_if_flip = gt_centline_3d.copy()

        optimized_parameter = np.load(optimized_para_path + '/para_history_frame_' + str(frame_id) + '.npy')

        # pt0 = gt_centline_3d_if_flip[0, :].copy()
        # pt2 = gt_centline_3d_if_flip[-1, :].copy()
        # pt2[0] = pt0[0]
        # pt2[2] = pt0[2]
        # pt1 = (pt0 + pt2) / 2
        # radius_scale = np.array([20])

        pt0 = gt_centline_3d_if_flip[0, :].copy()
        pt1 = optimized_parameter[-1, 0:3].copy()
        pt2 = optimized_parameter[-1, 3:6].copy()
        radius_scale = optimized_parameter[-1, 6].copy()
        # pdb.set_trace()

        #### ===================================================================
        #      without radius + only end-effector
        #### ===================================================================
        #
        para_gt = torch.cat((torch.from_numpy(pt1), torch.from_numpy(pt2), torch.as_tensor(radius_scale).unsqueeze(0)))
        para_start = torch.from_numpy(pt0).to(gpu_or_cpu)

        # para_init = torch.cat((torch.from_numpy(pt1) + torch.rand(3) * 0., torch.from_numpy(pt2) + torch.rand(3) * 0.0)).to(gpu_or_cpu)
        para_init = torch.cat((torch.from_numpy(pt1), torch.from_numpy(pt2), torch.as_tensor(radius_scale).unsqueeze(0))).to(gpu_or_cpu)

        img_ref_rgb = cv2.imread(img_save_path)
        img_raw_rgb = cv2.imread(img_raw_path)
        # img_ref_rgb = cv2.resize(img_ref_rgb, (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
        img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_RGB2GRAY)
        ret, img_ref_thre = cv2.threshold(img_ref_gray.copy(), 245, 255, cv2.THRESH_BINARY_INV)

        img_ref_thre_inv = cv2.bitwise_not(img_ref_thre)
        img_ref_binary = np.where(img_ref_thre_inv == 255, 1, img_ref_thre_inv)

        diff_model = DiffOptimizeModel(para_init=para_init,
                                       para_start=para_start,
                                       radius_gt_3d=radius_gt_3d,
                                       image_ref=img_ref_binary,
                                       image_ref_rgb=img_ref_rgb,
                                       gt_centline_3d=gt_centline_3d_if_flip,
                                       cylinder_primitive_path=cylinder_primitive_path,
                                       cam_K=cam_K_mat,
                                       cam_RT_H=cam_RT_H_mat,
                                       selected_frame_id=selected_frame_id,
                                       gpu_or_cpu=gpu_or_cpu).to(gpu_or_cpu)

        ## in order to obtain all needed data
        one_step_loss = diff_model()

        bezier_surface_vertices = diff_model.bezier_surface_vertices_npy

        ##### =========================================================
        ##### =========================================================
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.scatter(bezier_surface_vertices[:, 0], bezier_surface_vertices[:, 1], bezier_surface_vertices[:, 2], marker='o', s=0.1, color='#874C62')
        ax.view_init(azim=-90, elev=-70)
        # ax.view_init(azim=0, elev=23)
        axisEqual3D(ax)
        ax.grid(True)
        ax.set_title('3D Surface Vertices')
        ax.set_xlim((-10, 50))
        ax.set_ylim((-30, 30))
        ax.set_zlim((30, 90))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.tight_layout()
        # plt.show()
        save_3d_surface_path = '/home/fei/icra2023_diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/version_video_allframe_3d_surface/3d_surface_finalstep_frame_' + str(
            frame_id) + '.png'
        fig.savefig(save_3d_surface_path, dpi=300)
        plt.close(fig)
        ##### =========================================================
        ##### =========================================================

        # fig, axes = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [1.0, 1.0, 1.2], 'height_ratios': [1]})
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        colormap = mpl.cm.gray
        colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax = axes.ravel()
        ax[0].imshow(cv2.cvtColor(img_raw_rgb, cv2.COLOR_BGR2RGB), aspect="auto")
        ### =============================
        # plot four points
        id_keypoint_skeleton = diff_model.ref_skeleton_selected_id_list
        ax[0].plot(diff_model.ref_skeleton[id_keypoint_skeleton, 0], diff_model.ref_skeleton[id_keypoint_skeleton, 1], linestyle='-', marker='o', color='#E94560', markersize=6)
        # plot end-effector point
        ax[0].plot(diff_model.ref_skeleton[-1, 0], diff_model.ref_skeleton[-1, 1], marker='o', color='#FFDE00', markersize=6)
        ### =============================
        ax[0].set_title('Each Frame Reference Image')
        ax[0].axis('off')

        ax[1].imshow(diff_model.img_render_diffable, cmap=colormap, norm=colormap_norm, aspect="auto")
        ### =============================
        id_keypoint_centerline = diff_model.centerline_selected_id_list
        ax[1].plot(diff_model.bezier_proj_img_npy[id_keypoint_centerline, 0],
                   diff_model.bezier_proj_img_npy[id_keypoint_centerline, 1],
                   linestyle='-',
                   marker='o',
                   color='#E94560',
                   markersize=6,
                   linewidth=1)
        ax[1].plot(diff_model.bezier_proj_img_npy[-1, 0], diff_model.bezier_proj_img_npy[-1, 1], marker='o', color='#FFDE00', markersize=6)
        ### =============================
        ax[1].set_title('Each Frame Render Image')
        ax[1].axis('off')

        # plt.tight_layout()
        # plt.show()
        save_2d_render_path = '/home/fei/icra2023_diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/version_video_allframe_3d_surface/2d_render_finalstep_frame_' + str(
            frame_id) + '.png'
        fig.savefig(save_2d_render_path, dpi=300)
        plt.close(fig)
