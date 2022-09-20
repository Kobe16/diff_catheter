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


class DoDiffOptimization(nn.Module):

    def __init__(self, para_init, para_gt, diff_model, total_itr_steps, img_raw_rgb, data_frame_id, if_print_log=1):
        super(DoDiffOptimization, self).__init__()

        self.para_init = para_init
        self.if_print_log = if_print_log
        self.diff_model = diff_model
        self.total_itr_steps = total_itr_steps

        self.optimizer = torch.optim.Adam([self.para_init], lr=0.2)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.6)

        self.saved_para_history = np.zeros((1, self.para_init.shape[0]))
        self.saved_loss_history = np.array([999999.0])
        self.GD_Iteration = 0
        self.loss = None

        self.para_gt = para_gt

        self.img_raw_rgb = img_raw_rgb
        self.data_frame_id = data_frame_id

        self.save_data_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_only/frame_' + str(self.data_frame_id)
        Path(self.save_data_path).mkdir(parents=True, exist_ok=True)

    def savingStepFigures(self, save_img_path):

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        colormap = mpl.cm.gray
        colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax = axes.ravel()
        ax[0].imshow(cv2.cvtColor(self.img_raw_rgb, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Ref raw')
        ax[0].axis('off')

        ax[1].imshow(self.diff_model.image_ref.cpu().detach().numpy().copy(), cmap=colormap, norm=colormap_norm)
        ax[1].set_title('Ref mask')
        ax[1].axis('off')

        ax[2].imshow(self.img_render_init, cmap=colormap, norm=colormap_norm)
        ax[2].set_title('Initial render')
        ax[2].axis('off')

        ax[3].imshow(self.diff_model.img_render_diffable, cmap=colormap, norm=colormap_norm)
        ax[3].set_title('Final render')
        ax[3].axis('off')

        # ax[2].imshow(img_render_alpha.cpu().detach().numpy(), cmap=colormap, norm=colormap_norm)
        # ax[2].set_title('raw render')
        # ax[3].imshow(img_diff.cpu().detach().numpy(), cmap=colormap, norm=colormap_norm)
        # ax[3].set_title('difference')
        plt.tight_layout()
        # plt.show()
        fig.savefig(save_img_path, dpi=300)
        plt.close(fig)

        print('===============================================')
        print(self.diff_model.bezier_proj_img_npy[-1, :])
        print(self.diff_model.ref_skeleton[-1, :])
        print('===============================================')

        # pdb.set_trace()

    def savingFinalStepFigures(self, save_final_step_img_path, save_render_final_image_path):

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        colormap = mpl.cm.gray
        colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax = axes.ravel()
        ax[0].imshow(self.diff_model.img_render_diffable, cmap=colormap, norm=colormap_norm)
        ax[0].set_title('Mask')
        ax[0].axis('off')

        # ax[1].imshow(self.diff_model.img_render_contour, cmap=colormap, norm=colormap_norm)
        # ax[1].set_title('Contour')
        ax[1].imshow(self.diff_model.img_render_centerline, cmap=colormap, norm=colormap_norm)
        ax[1].set_title('Centerline Projection')
        ax[1].axis('off')

        ax[2].imshow(self.diff_model.img_render_diffable, cmap=colormap, norm=colormap_norm)
        ax[2].set_title('Keypoints (end-effector)')
        # plot end effector
        ax[2].plot(self.diff_model.bezier_proj_img_npy[-1, 0], self.diff_model.bezier_proj_img_npy[-1, 1], linestyle='-', marker='o', color='#FFDE00', markersize=6, linewidth=1)
        ax[2].axis('off')

        ax[3].imshow(self.diff_model.img_render_diffable, cmap=colormap, norm=colormap_norm)
        ax[3].set_title('Keypoints (4 points)')
        # plot four points
        id_keypoint_centerline = self.diff_model.centerline_selected_id_list
        ax[3].plot(self.diff_model.bezier_proj_img_npy[id_keypoint_centerline, 0], self.diff_model.bezier_proj_img_npy[id_keypoint_centerline, 1], marker='o', color='#E94560', markersize=6)
        ax[3].axis('off')

        # ax[2].plot(np.asarray([self.diff_model.bezier_proj_img_npy[0, 0], self.diff_model.pt_intesection_endpoints_render[0]]),
        #            np.asarray([self.diff_model.bezier_proj_img_npy[0, 1], self.diff_model.pt_intesection_endpoints_render[1]]),
        #            linestyle='-',
        #            marker='o',
        #            color='#5F6F94',
        #            markersize=6,
        #            linewidth=1)
        # ax[3].plot(np.asarray([self.diff_model.bezier_proj_img_npy[-1, 0], self.diff_model.pt_intesection_endpoints_render[0]]),
        #            np.asarray([self.diff_model.bezier_proj_img_npy[-1, 1], self.diff_model.pt_intesection_endpoints_render[1]]),
        #            linestyle='-',
        #            marker='o',
        #            color='#5F6F94',
        #            markersize=6,
        #            linewidth=1)

        plt.tight_layout()
        # plt.show()
        fig.savefig(save_final_step_img_path, dpi=300)
        plt.close(fig)

        ##### =========================================================
        ##### =========================================================
        color_red = (96, 69, 233)
        color_yellow = (0, 222, 255)
        color_green = (97, 131, 61)

        single_channel_render = self.diff_model.img_render_diffable.copy()
        single_channel_render = np.where(single_channel_render >= 0.1, 255, single_channel_render)
        single_channel_render = np.where(single_channel_render < 0.1, 0, single_channel_render)

        render_final_image = np.stack((single_channel_render, single_channel_render, single_channel_render), axis=2)

        for i in range(len(id_keypoint_centerline) - 1):
            id = id_keypoint_centerline[i]
            id_next = id_keypoint_centerline[i + 1]
            p1 = (int(self.diff_model.bezier_proj_img_npy[id, 0]), int(self.diff_model.bezier_proj_img_npy[id, 1]))
            p2 = (int(self.diff_model.bezier_proj_img_npy[id_next, 0]), int(self.diff_model.bezier_proj_img_npy[id_next, 1]))
            cv2.line(render_final_image, p1, p2, color_green, 1)

            cv2.circle(render_final_image, p1, radius=6, color=color_red, thickness=-1)

        ## draw endpoint
        cv2.circle(render_final_image, (int(self.diff_model.bezier_proj_img_npy[-1, 0]), int(self.diff_model.bezier_proj_img_npy[-1, 1])), radius=6, color=color_yellow, thickness=-1)

        # pdb.set_trace()

        cv2.imwrite(save_render_final_image_path, render_final_image)

        # pdb.set_trace()

    def savingReferenceFigures(self, save_img_path):

        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        colormap = mpl.cm.gray
        colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax = axes.ravel()
        ax[0].imshow(cv2.cvtColor(self.img_raw_rgb, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Ref image')
        ax[0].axis('off')

        ax[1].imshow(self.diff_model.image_ref_npy, cmap=colormap, norm=colormap_norm)
        ax[1].plot(self.diff_model.ref_skeleton[:, 0], self.diff_model.ref_skeleton[:, 1], linestyle='-', marker='o', color='#181818', markersize=0.2, linewidth=0.1)
        ax[1].set_title('Mask/skeleton')
        ax[1].axis('off')

        ax[2].imshow(self.diff_model.image_ref_npy, cmap=colormap, norm=colormap_norm)
        ax[2].set_title('Keypoints/endpoint')
        # ax[2].plot(self.diff_model.ref_skeleton[-1, 0], self.diff_model.ref_skeleton[-1, 1], linestyle='-', marker='o', color='#E94560', markersize=6, linewidth=1)
        # plot four points
        id_keypoint_skeleton = self.diff_model.ref_skeleton_selected_id_list
        ax[2].plot(self.diff_model.ref_skeleton[id_keypoint_skeleton, 0], self.diff_model.ref_skeleton[id_keypoint_skeleton, 1], linestyle='-', marker='o', color='#E94560', markersize=6)
        # plot end-effector point
        ax[2].plot(self.diff_model.ref_skeleton[-1, 0], self.diff_model.ref_skeleton[-1, 1], marker='o', color='#FFDE00', markersize=6)
        ax[2].axis('off')

        # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        # colormap = mpl.cm.binary
        # colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        # ax = axes.ravel()
        # ax[0].imshow(self.diff_model.image_ref_npy, cmap=colormap, norm=colormap_norm)
        # ax[0].set_title('Ref  mask')

        # ax[1].imshow(self.diff_model.image_ref_npy, cmap=colormap, norm=colormap_norm)
        # ax[1].plot(self.diff_model.ref_skeleton[:, 0], self.diff_model.ref_skeleton[:, 1], linestyle='-', marker='o', color='#F9F9F9', markersize=0.2, linewidth=0.1)
        # ax[1].set_title('Ref skeletonization')

        # ax[2].imshow(self.diff_model.image_ref_npy, cmap=colormap, norm=colormap_norm)
        # ax[2].set_title('Ref keypoints (end-effector)')
        # ax[2].plot(self.diff_model.ref_skeleton[-1, 0], self.diff_model.ref_skeleton[-1, 1], linestyle='-', marker='o', color='#E94560', markersize=6, linewidth=1)

        # ax[3].imshow(self.diff_model.image_ref_npy, cmap=colormap, norm=colormap_norm)
        # ax[3].set_title('Ref keypoints (4 points)')
        # # plot four points
        # id_keypoint_skeleton = self.diff_model.ref_skeleton_selected_id_list
        # ax[3].plot(self.diff_model.ref_skeleton[id_keypoint_skeleton, 0], self.diff_model.ref_skeleton[id_keypoint_skeleton, 1], linestyle='-', marker='o', color='#E94560', markersize=6)

        # ax[2].imshow(self.diff_model.image_ref_npy, cmap=colormap, norm=colormap_norm)
        # ax[2].set_title('Ref keypoints (end-effector)')
        # ax[2].plot(
        #     np.asarray([self.diff_model.ref_skeleton[0, 0], self.diff_model.pt_intesection_endpoints_ref[0]]),
        #     np.asarray([self.diff_model.ref_skeleton[0, 1], self.diff_model.pt_intesection_endpoints_ref[1]]),
        #     linestyle='-',
        #     #    marker='o',
        #     color='#5F6F94',
        #     # markersize=6,
        #     linewidth=1)
        # ax[2].plot(self.diff_model.ref_skeleton[0, 0], self.diff_model.ref_skeleton[0, 1], marker='o', color='#E94560', markersize=6)  ## start point
        # ax[2].plot(np.asarray([self.diff_model.ref_skeleton[-1, 0], self.diff_model.pt_intesection_endpoints_ref[0]]),
        #            np.asarray([self.diff_model.ref_skeleton[-1, 1], self.diff_model.pt_intesection_endpoints_ref[1]]),
        #            linestyle='-',
        #            marker='o',
        #            color='#5F6F94',
        #            markersize=6,
        #            linewidth=1)

        plt.tight_layout()
        plt.axis('off')
        # plt.show()
        fig.savefig(save_img_path, dpi=300)
        plt.close(fig)

        # pdb.set_trace()

    def doOptimization(self):

        def closure():
            # with autograd.detect_anomaly():
            self.optimizer.zero_grad()
            self.loss = self.diff_model()
            self.loss.backward()

            return self.loss

        self.optimizer.zero_grad()
        loss_history = []
        learn_rate_history = []
        # last_loss = 99999.0  # current loss value
        # last_learn_rate = 0.0  # current loss value

        converge = False  # converge or not
        self.id_iteration = 0  # number of updates

        # while not converge and self.id_iteration < self.total_itr_steps:
        while self.id_iteration < self.total_itr_steps:

            self.optimizer.step(closure)
            self.lr_scheduler.step()

            learn_rate = self.optimizer.param_groups[0]["lr"]

            # if (abs(self.loss - last_loss) < 1e-6):
            #     converge = True

            self.id_iteration += 1

            if self.if_print_log:
                print("Curr grad : ", self.para_init.grad)
                print("Curr para : ", self.para_init)
                print("Curr loss : ", self.loss)
                print("---------------- FINISH ", self.id_iteration, " ^_^ ITER ---------------- \n")

            save_img_steps_path = self.save_data_path + '/step_' + str(self.id_iteration) + '.png'

            if self.id_iteration >= self.total_itr_steps:
                save_final_step_img_path = self.save_data_path + '/final_frame_' + str(self.data_frame_id) + '.png'
                save_render_final_image_path = self.save_data_path + '/render_final_frame_' + str(self.data_frame_id) + '.png'
                self.savingFinalStepFigures(save_final_step_img_path, save_render_final_image_path)

                save_render_2d_skeleton_path = self.save_data_path + '/render_2d_skeleton_frame_' + str(self.data_frame_id) + '.npy'
                np.save(save_render_2d_skeleton_path, self.diff_model.bezier_proj_img_npy)

                save_render_3d_centerline_path = self.save_data_path + '/render_3d_centerline_frame_' + str(self.data_frame_id) + '.npy'
                np.save(save_render_3d_centerline_path, self.diff_model.bezier_pos_npy)

            if self.id_iteration == 1:
                self.img_render_init = self.diff_model.img_render_diffable.copy()

                save_img_ref_path = self.save_data_path + '/ref_frame_' + str(self.data_frame_id) + '.png'
                self.savingReferenceFigures(save_img_ref_path)

                save_ref_2d_skeleton_path = self.save_data_path + '/ref_2d_skeleton_frame_' + str(self.data_frame_id) + '.npy'
                np.save(save_ref_2d_skeleton_path, self.diff_model.ref_skeleton)

                save_ref_3d_centerline_path = self.save_data_path + '/ref_3d_centerline_frame_' + str(self.data_frame_id) + '.npy'
                np.save(save_ref_3d_centerline_path, self.diff_model.gt_centline_3d)

            ## save all steps
            self.savingStepFigures(save_img_steps_path)

            self.saved_loss_history = np.hstack((self.saved_loss_history, self.loss.cpu().detach().numpy()))
            self.saved_para_history = np.vstack((self.saved_para_history, self.para_init.cpu().detach().numpy()))

            # last_loss = torch.clone(self.loss)
            loss_history.append(self.loss.cpu().detach().numpy())
            learn_rate_history.append(learn_rate)

            # saved_value = np.hstack((last_loss.cpu().detach().numpy(), self.para_init.cpu().detach().numpy()))
            # save_mesh_path = '/home/fei/diff_catheter/scripts/diff_render/torch3d_rendered_imgs/meshes' + 'mesh_' + str(
            #     self.id_iteration) + '.obj'  # save the figure to file
            # self.diff_model.saveUpdatedMesh(save_mesh_path)

        print("Final --->", self.para_init.cpu().detach())
        print("GT    --->", self.para_gt)
        print("Error --->", torch.abs(self.para_init.cpu().detach() - self.para_gt))

        save_loss_history_path = self.save_data_path + '/loss_history_frame_' + str(self.data_frame_id) + '.npy'
        saved_para_history_path = self.save_data_path + '/para_history_frame_' + str(self.data_frame_id) + '.npy'
        np.save(save_loss_history_path, self.saved_loss_history)
        np.save(saved_para_history_path, self.saved_para_history)

        # np.savetxt(self.save_dir + '/final_optimized_para.csv', self.saved_opt_history, delimiter=",")
        # # plt.plot(loss_history)
        # plt.plot(loss_history, marker='o', linestyle='-', linewidth=1, markersize=4)
        # plt.show()

        fig, axes = plt.subplots(2, 1, figsize=(6, 7))
        ax = axes.ravel()
        ax[0].plot(loss_history, marker='o', linestyle='-', linewidth=1, markersize=4)
        ax[0].set_title('Loss')

        ax[1].plot(learn_rate_history, marker='o', linestyle='-', linewidth=1, markersize=4)
        ax[1].set_title('Learning Rate')

        # plt.show()
        plt.tight_layout()
        plt.close(fig)
        save_loss_history_path = self.save_data_path + '/loss_history_frame_' + str(self.data_frame_id) + '.png'
        fig.savefig(save_loss_history_path, dpi=300)

        # return self.saved_opt_history, self.para


if __name__ == '__main__':

    for i in range(8, 16):

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
        frame_naming = sim_case + 'left_recif_binary_' + str(frame_id)
        img_save_path = frame_naming + '.jpg'
        img_raw_path = sim_case + 'left_recif_raw_' + str(frame_id) + '.jpg'
        cc_specs_path = frame_naming + '.npy'
        target_specs_path = None
        viewpoint_mode = 1
        transparent_mode = 0

        cylinder_primitive_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/shape_primitive/cylinder_primitve_101_40.obj'

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

        pt0 = gt_centline_3d_if_flip[0, :].copy()
        pt2 = gt_centline_3d_if_flip[-1, :].copy()
        pt2[0] = pt0[0]
        pt2[2] = pt0[2]
        pt1 = (pt0 + pt2) / 2

        radius_scale = np.array([20])

        # pdb.set_trace()

        # # each centerline_gt has 31 values, radius_gt has 30 values. we made them into the same size : 30
        # dataset = DatasetProcess(centerline_gt[:, :, 0:-1], radius_gt[:, :], frame_id)
        # # (pt0, pt1, pt2, pt3) = dataset.get_initial_guess_cubic_bezier()
        # (pt0, pt1, pt2) = dataset.get_initial_guess_quadratic_bezier(frame_id=10)
        # # (pt0, pt1, pt2) = dataset.get_initial_guess_quadratic_bezier_frame0()

        #### ===================================================================
        #      without radius + only end-effector
        #### ===================================================================
        #
        para_gt = torch.cat((torch.from_numpy(pt1), torch.from_numpy(pt2), torch.from_numpy(radius_scale)))
        para_start = torch.from_numpy(pt0).to(gpu_or_cpu)

        # para_init = torch.cat((torch.from_numpy(pt1) + torch.rand(3) * 0., torch.from_numpy(pt2) + torch.rand(3) * 0.0)).to(gpu_or_cpu)
        para_init = torch.cat((torch.from_numpy(pt1), torch.from_numpy(pt2), torch.from_numpy(radius_scale))).to(gpu_or_cpu)

        # init_random = torch.tensor([0.5799, 0.0191, 0.0749, 0.7582, 0.0996, 0.0509], dtype=torch.float).to(gpu_or_cpu)
        # init_random = torch.tensor([-8.00, 0.0, 1.5, 1.000, 0.0, 1.5], dtype=torch.float).to(gpu_or_cpu)

        #### ===================================================================

        # used before frame 117 : for mask+4k
        # init_random = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)

        # frame 93/94
        # init_random = torch.tensor([5.0, 4.0, 0.0, 10.0, 0.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)

        # frame 105/104
        # init_random = torch.tensor([5.0, 10.0, 0.0, 50.0, 5.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)

        # from frame 109--121
        init_random = torch.tensor([8.0, 5.0, 0.0, 18.0, 5.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)

        # ## from frame 117--121 : weight to 100
        # init_random = torch.tensor([-2.0, 10.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)

        ## from frame 121--125 : weight to 100
        # init_random = torch.tensor([2.0, 5.0, 0.0, -2.0, 0.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)
        ## for mask only
        # init_random = torch.tensor([-1.0, 2.0, 0.0, -8.0, 0.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)

        ## from frame 135--end : weight to 100
        # init_random = torch.tensor([0.0, 5.0, 0.0, -10.0, 0.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu)
        # init_random = torch.tensor([-4.0, 10.0, 0.0, -15.0, 0.0, 0.0, 0.0], dtype=torch.float).to(gpu_or_cpu) ## good for all

        # para_init = para_init
        para_init = para_init + init_random

        para_init.requires_grad = True

        # pdb.set_trace()

        #### ===================================================================
        #      without radius
        #### ===================================================================
        #
        # para_gt = torch.cat((torch.from_numpy(pt0), torch.from_numpy(pt1), torch.from_numpy(pt2)))
        # para_init = torch.cat((torch.from_numpy(pt0) + torch.rand(3) * 0.1, torch.from_numpy(pt1) + torch.rand(3) * 0.1, torch.from_numpy(pt2) + torch.rand(3) * 0.1)).to(gpu_or_cpu)

        #### ===================================================================
        #      including radius
        #### ===================================================================
        # para_gt = torch.cat((torch.from_numpy(pt0), torch.from_numpy(pt1), torch.from_numpy(pt2),
        #                      torch.from_numpy(np.asarray([radius_gt_3d[0], radius_gt_3d[-1]]))))
        # para_init = torch.cat((torch.from_numpy(pt0) + torch.rand(3) * 1, torch.from_numpy(pt1) + torch.rand(3) * 1,
        #                        torch.from_numpy(pt2) + torch.rand(3) * 1,
        #                        torch.from_numpy(np.asarray([radius_gt_3d[0], radius_gt_3d[0]])))).to(gpu_or_cpu)
        # para_init = torch.tensor([
        #     -1.3260e+00, 3.4619e+00, -9.6466e-01, 1.4029e+01, 1.2546e+00, 6.1034e-01, 1.4239e+01, 3.3831e-03, -4.6002e+00,
        #     9.9970e-01, 9.9970e-01
        # ],
        #                          dtype=torch.float).to(gpu_or_cpu)

        #### ===================================================================
        #### ===================================================================

        total_itr_steps = 200

        # pdb.set_trace()

        img_ref_rgb = cv2.imread(img_save_path)
        img_raw_rgb = cv2.imread(img_raw_path)
        # img_ref_rgb = cv2.resize(img_ref_rgb, (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
        img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_RGB2GRAY)
        ret, img_ref_thre = cv2.threshold(img_ref_gray.copy(), 245, 255, cv2.THRESH_BINARY_INV)

        img_ref_thre_inv = cv2.bitwise_not(img_ref_thre)
        img_ref_binary = np.where(img_ref_thre_inv == 255, 1, img_ref_thre_inv)
        # img_ref_binary = np.where(img_ref_thre == 255, 1, img_ref_thre)

        # # ---------------
        # # plot with
        # # ---------------
        # fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        # ax = axes.ravel()

        # ax[0].imshow(cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2RGB))
        # ax[0].set_title('2d img_ref_rgb')

        # ax[1].imshow(img_ref_binary)
        # ax[1].set_title('2d img_ref_binary')

        # plt.tight_layout()
        # plt.show()

        # pdb.set_trace()

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

        do_diff = DoDiffOptimization(para_init=diff_model.para_init,
                                     para_gt=para_gt,
                                     diff_model=diff_model,
                                     total_itr_steps=total_itr_steps,
                                     img_raw_rgb=img_raw_rgb,
                                     data_frame_id=frame_id,
                                     if_print_log=1)

        do_diff.doOptimization()

    ###========================================================
    ### Render catheter using Blender
    ###========================================================
    # raw_render_catheter = BlenderRenderCatheter()
    # raw_render_catheter.set_bezier_in_blender(para_gt.detach().numpy(), p_start.detach().numpy())

    # raw_render_catheter.render_bezier_in_blender(cc_specs_path, img_save_path, target_specs_path, viewpoint_mode,
    #                                           transparoptimizerent_mode)

    # ##========================================================
    # ## Build Bezier Suface Mesh
    # ##========================================================
    # build_octupus = ConstructionBezier()

    # ## define bezier radius
    # build_octupus.getBezierRadius(para_init[9], para_init[10])

    # ## define a bezier curve
    # build_octupus.getBezierCurve(para_init[3:9], p_start=para_init[0:3])

    # ## get the bezier in TNB frame, in order to build a tube mesh
    # build_octupus.getBezierTNB(build_octupus.bezier_pos_cam, build_octupus.bezier_der_cam,
    #                            build_octupus.bezier_snd_der_cam)
    # # build_octupus.getBezierTNB(build_octupus.bezier_pos, build_octupus.bezier_der, build_octupus.bezier_snd_der)

    # # ## get bezier surface mesh
    # # ## ref : https://mathworld.wolfram.com/Tube.html
    # build_octupus.getBezierSurface(build_octupus.bezier_pos_cam)
    # # build_octupus.getBezierSurface(build_octupus.bezier_pos)

    # # build_octupus.createCylinderPrimitive()
    # # # build_octupus.createOpen3DVisualizer()
    # # build_octupus.updateOpen3DVisualizer()

    # # ## load the raw RGB image
    # build_octupus.loadRawImage(img_save_path)

    # # the following line is already implemented in "getBezierCurve" func
    # # build_octupus.proj_bezier_img = build_octupus.getProjPointCam(build_octupus.bezier_pos_cam, build_octupus.cam_K)

    # build_octupus.draw2DCenterlineImage()

    # pdb.set_trace()

    ###========================================================
    ### Differentiable Rendering
    ###========================================================
    # torch3d_render_catheter = DiffRenderCatheter(build_octupus.cam_RT_H, build_octupus.cam_K)
    # torch3d_render_catheter.loadCylinderPrimitive(cylinder_primitive_path)

    # torch3d_render_catheter.updateCylinderPrimitive(build_octupus.updated_surface_vertices)

    # img_id = 0
    # save_img_path = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/torch3d_rendered_imgs/' + 'torch3d_render_' + str(
    #     img_id) + '.jpg'  # save the figure to file
    # # torch3d_render_catheter.renderDeformedMesh(save_img_path)