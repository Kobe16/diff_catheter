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

# loss_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/'
# loss_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_end/'
loss_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_only/'

# frame_id = (93, 105, 117, 136)

frame_id = (93, 94, 95, 97, 98, 104, 105, 108, 109, 110, 111, 112, \
                113, 114, 115, 116, 117, 118, 119, 120, \
                121, 122, 123, 124, 125, 135, 136, 137, \
                138, 139, 140, 147)
# frame_id = (93, 94, 95, 97, 98, 105, 108, 109, 110, 111, 112, \
#                 113, 114, 115, 116, 117, 118, 119, 120, \
#                 121, 122, 123, 124, 125, 135, 136, 137, \
#                 138, 139, 140, 147)
error_2d = []
error_3d = []

for i in range(len(frame_id)):

    ref_3d_centerline_path = dataset_path + 'centerlines_denosie/centerline_denosie_' + str(frame_id[i]) + '.npy'
    ref_2d_skeleton_path = loss_path + 'frame_' + str(frame_id[i]) + '/ref_2d_skeleton_frame_' + str(frame_id[i]) + '.npy'
    ref_3d_centerline = np.load(ref_3d_centerline_path)
    ref_2d_skeleton = np.load(ref_2d_skeleton_path)

    render_3d_centerline_path = loss_path + 'frame_' + str(frame_id[i]) + '/render_3d_centerline_frame_' + str(frame_id[i]) + '.npy'
    render_2d_skeleton_path = loss_path + 'frame_' + str(frame_id[i]) + '/render_2d_skeleton_frame_' + str(frame_id[i]) + '.npy'
    render_3d_centerline = np.load(render_3d_centerline_path)
    render_2d_skeleton = np.load(render_2d_skeleton_path)

    img_raw_path = dataset_path + 'left_recif_raw_' + str(frame_id[i]) + '.jpg'
    img_raw_rgb = cv2.imread(img_raw_path)

    img_render_binary_path = loss_path + 'frame_' + str(frame_id[i]) + '/render_final_frame_' + str(frame_id[i]) + '.png'
    img_render_binary = cv2.imread(img_render_binary_path)

    ### 2D
    diff_render_2d_skeleton = np.diff(render_2d_skeleton, axis=0)
    dis_diff_render_2d_skeleton = np.linalg.norm(diff_render_2d_skeleton, ord=None, axis=1, keepdims=False)
    dis_sum_render_2d_skeleton = np.cumsum(dis_diff_render_2d_skeleton, axis=0)
    dis_sum_render_2d_skeleton = np.hstack((np.array([0]), dis_sum_render_2d_skeleton))
    dis_sum_render_2d_skeleton = dis_sum_render_2d_skeleton / dis_sum_render_2d_skeleton[-1]

    diff_ref_2d_skeleton = np.diff(ref_2d_skeleton, axis=0)
    dis_diff_ref_2d_skeleton = np.linalg.norm(diff_ref_2d_skeleton, ord=None, axis=1, keepdims=False)
    dis_sum_ref_2d_skeleton = np.cumsum(dis_diff_ref_2d_skeleton, axis=0)
    dis_sum_ref_2d_skeleton = np.hstack((np.array([0]), dis_sum_ref_2d_skeleton))
    dis_sum_ref_2d_skeleton = dis_sum_ref_2d_skeleton / dis_sum_ref_2d_skeleton[-1]

    if dis_sum_render_2d_skeleton.shape[0] > dis_sum_ref_2d_skeleton.shape[0]:
        print("this frame has error : ", i)
    else:
        select_ref_2d_skeleton_by_dis = []
        for i in range(dis_sum_render_2d_skeleton.shape[0]):
            err = np.abs(dis_sum_render_2d_skeleton[i] - dis_sum_ref_2d_skeleton)
            index = np.argmin(err)
            temp = ref_2d_skeleton[index, :]
            select_ref_2d_skeleton_by_dis.append(temp)
        select_ref_2d_skeleton_by_dis = np.stack(select_ref_2d_skeleton_by_dis)

        error_2d_skeleton_by_dis = select_ref_2d_skeleton_by_dis - render_2d_skeleton
        norm_error_2d_skeleton_by_dis = np.linalg.norm(error_2d_skeleton_by_dis, ord=None, axis=1, keepdims=False)

        error_2d.append(np.sum(norm_error_2d_skeleton_by_dis) / error_2d_skeleton_by_dis.shape[0])

    ### 2D
    diff_render_3d_centerline = np.diff(render_3d_centerline, axis=0)
    dis_diff_render_3d_centerline = np.linalg.norm(diff_render_3d_centerline, ord=None, axis=1, keepdims=False)
    dis_sum_render_3d_centerline = np.cumsum(dis_diff_render_3d_centerline, axis=0)
    dis_sum_render_3d_centerline = np.hstack((np.array([0]), dis_sum_render_3d_centerline))
    dis_sum_render_3d_centerline = dis_sum_render_3d_centerline / dis_sum_render_3d_centerline[-1]

    diff_ref_3d_centerline = np.diff(ref_3d_centerline, axis=0)
    dis_diff_ref_3d_centerline = np.linalg.norm(diff_ref_3d_centerline, ord=None, axis=1, keepdims=False)
    dis_sum_ref_3d_centerline = np.cumsum(dis_diff_ref_3d_centerline, axis=0)
    dis_sum_ref_3d_centerline = np.hstack((np.array([0]), dis_sum_ref_3d_centerline))
    dis_sum_ref_3d_centerline = dis_sum_ref_3d_centerline / dis_sum_ref_3d_centerline[-1]

    if dis_sum_render_3d_centerline.shape[0] > dis_sum_ref_3d_centerline.shape[0]:
        print("==============================================")
        print("this frame has error : ", i)
        print("==============================================")
    else:
        select_ref_3d_centerline_by_dis = []
        for i in range(dis_sum_render_3d_centerline.shape[0]):
            err = np.abs(dis_sum_render_3d_centerline[i] - dis_sum_ref_3d_centerline)
            index = np.argmin(err)
            temp = ref_3d_centerline[index, :]
            select_ref_3d_centerline_by_dis.append(temp)
        select_ref_3d_centerline_by_dis = np.stack(select_ref_3d_centerline_by_dis)

        error_3d_skeleton_by_dis = select_ref_3d_centerline_by_dis - render_3d_centerline
        norm_error_3d_skeleton_by_dis = np.linalg.norm(error_3d_skeleton_by_dis, ord=None, axis=1, keepdims=False)

        error_3d.append(np.sum(norm_error_3d_skeleton_by_dis) / error_3d_skeleton_by_dis.shape[0])

    # pdb.set_trace()

    # # ---------------
    # # plot with
    # # ---------------
    # fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    # ax = axes.ravel()

    # ax[0].imshow(cv2.cvtColor(img_raw_rgb, cv2.COLOR_BGR2RGB))
    # ax[0].set_title('reference raw')
    # ax[0].plot(ref_2d_skeleton[:, 0], ref_2d_skeleton[:, 1], linestyle='-', marker='o', color='#E94560', markersize=1.0, linewidth=0.1)
    # ax[0].plot(render_2d_skeleton[:, 0], render_2d_skeleton[:, 1], linestyle='-', marker='o', color='#3D8361', markersize=1.0, linewidth=0.1)
    # ax[0].axis('off')
    # ax[1].imshow(img_render_binary)

    # ax[1].set_title('final render')
    # ax[1].axis('off')

    # plt.tight_layout()
    # plt.show()

    # ---------------
    # plot with
    # ---------------
    # fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    # ax = axes.ravel()

    # ax[0].plot(ref_3d_centerline[:, 0], linestyle='-', marker='o', color='#E94560', markersize=1.0, linewidth=0.1)
    # ax[0].plot(render_3d_centerline[:, 0], linestyle='-', marker='o', color='#3D8361', markersize=1.0, linewidth=0.1)
    # ax[0].set_title('3d x coordinates')

    # ax[1].plot(ref_3d_centerline[:, 1], linestyle='-', marker='o', color='#E94560', markersize=1.0, linewidth=0.1)
    # ax[1].plot(render_3d_centerline[:, 1], linestyle='-', marker='o', color='#3D8361', markersize=1.0, linewidth=0.1)
    # ax[1].set_title('3d y coordinates')

    # ax[2].plot(ref_3d_centerline[:, 2], linestyle='-', marker='o', color='#E94560', markersize=1.0, linewidth=0.1)
    # ax[2].plot(render_3d_centerline[:, 2], linestyle='-', marker='o', color='#3D8361', markersize=1.0, linewidth=0.1)
    # ax[2].set_title('3d z coordinates')

    # plt.tight_layout()
    # plt.show()

error_2d = np.stack(error_2d)
std_error_2d = np.std(error_2d)
average_error_2d = np.sum(error_2d) / len(frame_id)
print("==============================================")
print('2d average error is : ', average_error_2d)
print('all 2d error     is : ', error_2d)
print('std_error_2d     is : ', std_error_2d)
print("==============================================")
print("\n")

error_3d = np.stack(error_3d)
std_error_3d = np.std(error_3d)
average_error_3d = np.sum(error_3d) / len(frame_id)
print("==============================================")
print('3d average error is : ', average_error_3d)
print('all 3d error     is : ', error_3d)
print('std_error_3d     is : ', std_error_3d)
print("==============================================")