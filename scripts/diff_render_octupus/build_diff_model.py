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

import open3d as o3d

import cv2
import matplotlib.pyplot as plt

import pdb

from construction_bezier import ConstructionBezier
from blender_catheter import BlenderRenderCatheter
from diff_render_catheter import DiffRenderCatheter
from loss_define import ContourLoss, MaskLoss, CenterlineLoss, KeypointsInImageLoss, KeypointsIn3DLoss

import pytorch3d
import pytorch3d.io as torch3d_io

import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl

from tqdm.notebook import tqdm

import skfmm


class DiffOptimizeModel(nn.Module):

    def __init__(self, para_init, para_start, radius_gt_3d, image_ref, image_ref_rgb, gt_centline_3d, cylinder_primitive_path, cam_K, cam_RT_H, selected_frame_id, gpu_or_cpu):
        super().__init__()

        self.gpu_or_cpu = gpu_or_cpu

        self.build_bezier = ConstructionBezier(cam_K, cam_RT_H, gpu_or_cpu)
        self.build_bezier.to(gpu_or_cpu)

        self.torch3d_render_catheter = DiffRenderCatheter(self.build_bezier.cam_RT_H, self.build_bezier.cam_K, gpu_or_cpu)
        self.torch3d_render_catheter.to(gpu_or_cpu)
        self.torch3d_render_catheter.loadCylinderPrimitive(cylinder_primitive_path)

        self.mask_loss = MaskLoss(device=gpu_or_cpu)
        self.mask_loss.to(gpu_or_cpu)

        self.contour_loss = ContourLoss(device=gpu_or_cpu)
        self.contour_loss.to(gpu_or_cpu)

        self.centerline_loss = CenterlineLoss(device=gpu_or_cpu)
        self.centerline_loss.to(gpu_or_cpu)

        self.keypoints_image_loss = KeypointsInImageLoss(device=gpu_or_cpu)
        self.keypoints_image_loss.to(gpu_or_cpu)

        self.gt_centline_3d = gt_centline_3d
        self.build_bezier.getGroundTruthCenterlineCam(gt_centline_3d)

        self.keypoints_3d_loss = KeypointsIn3DLoss(device=gpu_or_cpu)
        self.keypoints_3d_loss.to(gpu_or_cpu)

        ### Straight Line
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)

        self.para_init = para_init
        self.radius_gt_3d = radius_gt_3d

        # self.p_start = self.para_init[0:3]
        # self.p_mid_end = self.para_init[3:9]

        #### =====================================
        #      only end-effector
        #### =====================================
        self.p_start = para_start
        self.p_mid_end = self.para_init
        self.radius_scale = para_init[-1]

        #### =====================================
        #      including radius
        #### =====================================
        # self.radius_start = self.para_init[9]
        # self.radius_end = self.para_init[10]

        #### =====================================
        #      without radius
        #### =====================================
        self.radius_start = radius_gt_3d[0]
        self.radius_end = radius_gt_3d[-1]

        ### GT values
        # self.para_init = nn.Parameter(
        #     torch.from_numpy(
        #         np.array([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896],
        #                  dtype=np.float32)).to(gpu_or_cpu))

        # # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))

        # image_ref = torch.from_numpy(image_ref.astype(np.float32))
        # self.register_buffer('image_ref', image_ref)

        # pdb.set_trace()

        # img_ref_dist_map = skfmm.distance(image_ref)
        img_ref_dist_map = skfmm.distance(np.logical_not(image_ref).astype(int))
        self.img_ref_dist_map = torch.from_numpy(img_ref_dist_map).to(gpu_or_cpu)
        self.image_ref = torch.from_numpy(image_ref.astype(np.float32)).to(gpu_or_cpu)
        self.image_ref_rgb = image_ref_rgb
        # self.register_buffer('image_ref', image_ref)

        self.pt_intesection_endpoints_ref = None
        self.pt_intesection_endpoints_render = None

        self.selected_frame_id = selected_frame_id

    def forward(self, save_img_path=None):

        ###========================================================
        ### get Bezier Surface
        ###========================================================
        ## define bezier radius
        self.build_bezier.getBezierRadius(self.radius_start, self.radius_end, self.radius_scale)

        ## define a bezier curve
        self.build_bezier.getQuadraticBezierCurve(self.p_mid_end, self.p_start)
        # self.build_bezier.getCubicBezierCurve(self.p_mid_end, self.p_start)

        ## get the bezier in TNB frame, in order to build a tube mesh
        # self.build_bezier.getBezierTNB(self.build_bezier.bezier_pos_cam, self.build_bezier.bezier_der_cam,
        #                                self.build_bezier.bezier_snd_der_cam)
        self.build_bezier.getBezierTNB(self.build_bezier.bezier_pos, self.build_bezier.bezier_der, self.build_bezier.bezier_snd_der)

        ## get bezier surface mesh
        ## ref : https://mathworld.wolfram.com/Tube.html
        # self.build_bezier.getBezierSurface(self.build_bezier.bezier_pos_cam)
        self.build_bezier.getBezierSurface(self.build_bezier.bezier_pos)

        # self.build_bezier.createCylinderPrimitive()
        # # self.build_bezier.createOpen3DVisualizer()
        # self.build_bezier.updateOpen3DVisualizer()
        # pdb.set_trace()

        # self.build_bezier.draw2DCenterlineImage(self.image_ref_rgb)1

        ###========================================================
        ### Render Catheter Using PyTorch3D
        ###========================================================
        self.torch3d_render_catheter.updateCylinderPrimitive(self.build_bezier.updated_surface_vertices)
        self.torch3d_render_catheter.renderDeformedMesh(save_img_path)

        ###========================================================
        ### Loss ： different combinations
        ###========================================================
        # img_render = self.torch3d_render_catheter.render_catheter_img[0, ..., 3].unsqueeze(0).unsqueeze(0)
        # # loss = torch.mean((img_render - self.image_ref.unsqueeze(0).unsqueeze(0))**2)
        # loss = F.l1_loss(img_render, self.image_ref.unsqueeze(0).unsqueeze(0))
        # print(loss)
        # loss = torch.mean((img_render - self.image_ref.unsqueeze(0).unsqueeze(0))**2)

        # img_render_mask = self.torch3d_render_catheter.render_catheter_img[0, ..., 3]
        # max_img_render_alpha = torch.max(self.torch3d_render_catheter.render_catheter_img[0, ..., 3])
        # img_render_alpha_norm = img_render_mask / max_img_render_alpha
        # img_diff = torch.abs(img_render_alpha_norm - self.image_ref)

        ##### -----------------------------
        img_render_mask = self.torch3d_render_catheter.render_catheter_img[0, ..., 3]

        ##### -----------------------------
        ####  Contour Loss
        loss_contour, img_render_contour, img_render_diffable = self.contour_loss(img_render_mask.unsqueeze(0), self.image_ref.unsqueeze(0), self.img_ref_dist_map.unsqueeze(0))
        ##### -----------------------------
        ####  Mask Loss : using a differentiable binarized image
        loss_mask = self.mask_loss(img_render_mask, self.image_ref)
        img_diff = torch.abs(img_render_mask - self.image_ref)

        ##### -----------------------------
        #### Centerline Loss
        loss_centerline, ref_skeleton, centerline_selected_id_list, ref_skeleton_selected_id_list = self.centerline_loss(self.build_bezier.bezier_proj_img, self.image_ref, self.selected_frame_id)
        img_render_centerline = self.build_bezier.draw2DCenterlineImage(self.image_ref, img_render_diffable, ref_skeleton)

        ##### -----------------------------
        #### Keypoints Loss
        # loss_keypoints_3d = self.keypoints_3d_loss(self.build_bezier.bezier_pos, self.gt_centline_3d)

        # # ---------------
        # # # plot with
        # # # ---------------
        # fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        # ax = axes.ravel()
        # ax[0].imshow(cv2.cvtColor(self.image_ref.cpu().detach().numpy(), cv2.COLOR_BGR2RGB))
        # gt_centline_proj_img = self.build_bezier.gt_centline_proj_img.cpu().detach().numpy()
        # # ax[0].plot(gt_centline_proj_img[:, 0], gt_centline_proj_img[:, 1], marker='o', color='#E94560', markersize=0.1)
        # # ax[0].plot(gt_centline_proj_img[-5, 0], gt_centline_proj_img[-5, 1], marker='o', color='#3D8361', markersize=6)
        # # ax[0].plot(ref_skeleton[:, 0], ref_skeleton[:, 1], marker='o', color='#E94560', markersize=0.1, linestyle='-', linewidth=1)
        # ax[0].plot(ref_skeleton[10:-30, 0], ref_skeleton[10:-30, 1], marker='o', color='#E94560', markersize=0.1, linestyle='-', linewidth=1)
        # ax[0].set_title('2d image_ref')

        # ax[1].imshow(img_render_diffable.cpu().detach().numpy())
        # # ax[1].plot(self.build_bezier.bezier_proj_img[-1, :], marker='o', color='#E94560', markersize=6)
        # ax[1].set_title('2d img_render')
        # plt.tight_layout()
        # plt.show()

        # pdb.set_trace()

        # loss_keypoints_image, pt_intesection_endpoints_ref, pt_intesection_endpoints_render = self.keypoints_image_loss(self.build_bezier.bezier_proj_img, self.build_bezier.bezier_der_proj_img,
        #                                                                                                                 self.build_bezier.gt_centline_proj_img)
        self.ref_skeleton_torch = torch.from_numpy(ref_skeleton.astype(np.float32)).to(self.gpu_or_cpu)
        loss_keypoints_image, pt_intesection_endpoints_ref, pt_intesection_endpoints_render = self.keypoints_image_loss(self.build_bezier.bezier_proj_img, self.build_bezier.bezier_der_proj_img,
                                                                                                                        self.ref_skeleton_torch)

        # img_render_keypoints2d, img_ref_keypoints2d = self.build_bezier.draw2DKeyPointsImage(self.image_ref_rgb, img_render_diffable, ref_skeleton, pt_intesection_endpoints_ref,
        #                                                                                      pt_intesection_endpoints_render)

        # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        # colormap = mpl.cm.binary
        # colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        # ax = axes.ravel()
        # ax[0].imshow(self.image_ref.cpu().detach().numpy(), cmap=colormap, norm=colormap_norm)
        # ax[0].set_title('reference binary')
        # ax[1].imshow(img_render_diffable.cpu().detach().numpy(), cmap=colormap, norm=colormap_norm)
        # ax[1].set_title('render diffable')
        # ax[2].imshow(img_render_centerline, cmap=colormap, norm=colormap_norm)
        # ax[2].set_title('render centerline')
        # # im = ax[3].imshow(img_diff.cpu().detach().numpy(), cmap=colormap, norm=colormap_norm)
        # # im = ax[3].imshow(image_ref_keypoints2d, cmap=colormap, norm=colormap_norm)
        # # ax[3].set_title('difference')
        # # plt.colorbar(im, ax=axes.ravel().tolist())
        # plt.show()

        ### for debugging NAN value torch
        # loss = self.torch3d_render_catheter.render_catheter_img[0, ..., 0][1, 1]
        # loss = torch.sum(self.torch3d_render_catheter.render_catheter_img[0, ..., 3])
        # loss = self.torch3d_render_catheter.render_cameras.get_projection_transform().get_matrix()[0, 0, 0]
        # loss = self.torch3d_render_catheter.updated_cylinder_primitive_mesh.verts_list()[0][0, 0]
        # img_render_binary = None

        # weight = torch.tensor([0.0, 1.0, 0.0, 0.0])
        # loss = loss_contour * weight[0] + loss_mask * weight[1] + loss_centerline * weight[2] + loss_keypoints_image * weight[3]

        weight = torch.tensor([1.0, 100.0, 0.0])
        loss = loss_mask * weight[0] + loss_centerline * weight[1] + loss_keypoints_image * weight[2]

        print("------------------------------------------------")
        print("loss_contour     : ", loss_contour)
        print("loss_mask        : ", loss_mask)
        print("loss_centerline  : ", loss_centerline)
        print("loss_keypoints   : ", loss_keypoints_image)
        print("------------------------------------------------")

        # pdb.set_trace()

        self.img_render_mask = img_render_mask.cpu().detach().numpy()
        self.img_render_contour = img_render_contour.cpu().detach().numpy()
        self.img_render_diffable = img_render_diffable.cpu().detach().numpy()
        self.img_render_centerline = img_render_centerline
        self.ref_skeleton = ref_skeleton
        self.image_ref_npy = self.image_ref.cpu().detach().numpy()
        self.pt_intesection_endpoints_ref = pt_intesection_endpoints_ref.cpu().detach().numpy()
        self.pt_intesection_endpoints_render = pt_intesection_endpoints_render.cpu().detach().numpy()
        self.bezier_proj_img_npy = self.build_bezier.bezier_proj_img.cpu().detach().numpy()
        self.bezier_pos_npy = self.build_bezier.bezier_pos.cpu().detach().numpy()

        self.ref_skeleton_selected_id_list = ref_skeleton_selected_id_list
        self.centerline_selected_id_list = centerline_selected_id_list

        self.bezier_surface_vertices_npy = self.build_bezier.updated_surface_vertices.cpu().detach().numpy()

        # pdb.set_trace()

        return loss

    def saveUpdatedMesh(self, save_mesh_path=None):
        updated_verts = self.torch3d_render_catheter.updated_cylinder_primitive_mesh.verts_list()
        updated_faces = self.torch3d_render_catheter.updated_cylinder_primitive_mesh.faces_list()

        # pdb.set_trace()

        torch3d_io.save_obj(save_mesh_path, updated_verts[0], updated_faces[0])