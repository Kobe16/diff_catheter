import sys
from turtle import pd

sys.path.append('..')

import os
import numpy as np

# import transforms
# import bezier_interspace_transforms
from bezier_set import BezierSet
import camera_settings

import torch

import open3d as o3d

import cv2
import matplotlib.pyplot as plt

import pdb

from construction_bezier import ConstructionBezier
from blender_catheter import BlenderRenderCatheter
from diff_render_catheter import DiffRenderCatheter
from loss_define import ContourLoss, MaskLoss

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap

from tqdm.notebook import tqdm


class DiffOptimizeModel(nn.Module):

    def __init__(self, para_init, p_start, image_ref, cylinder_primitive_path, gpu_or_cpu):
        super().__init__()

        self.build_bezier = ConstructionBezier()
        self.build_bezier.to(gpu_or_cpu)

        self.torch3d_render_catheter = DiffRenderCatheter(self.build_bezier.cam_RT_H, self.build_bezier.cam_K,
                                                          gpu_or_cpu)
        self.torch3d_render_catheter.to(gpu_or_cpu)
        self.torch3d_render_catheter.loadCylinderPrimitive(cylinder_primitive_path)

        self.mask_loss = MaskLoss(device=gpu_or_cpu)
        self.mask_loss.to(gpu_or_cpu)

        self.p_start = p_start.to(gpu_or_cpu)

        ### Straight Line
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)

        self.para_init = para_init

        ### GT values
        # self.para_init = nn.Parameter(
        #     torch.from_numpy(
        #         np.array([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896],
        #                  dtype=np.float32)).to(gpu_or_cpu))

        # # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))

        # image_ref = torch.from_numpy(image_ref.astype(np.float32))
        # self.register_buffer('image_ref', image_ref)
        self.image_ref = torch.from_numpy(image_ref.astype(np.float32)).to(gpu_or_cpu)
        # self.register_buffer('image_ref', image_ref)

    def forward(self, save_img_path=None):

        ###========================================================
        ### get Bezier Surface
        ###========================================================
        ## define a bezier curve
        self.build_bezier.getBezierCurve(self.para_init, self.p_start)
        ## get the bezier in TNB frame, in order to build a tube mesh
        # build_bezier.getBezierTNB(build_bezier.bezier_pos_cam, build_bezier.bezier_der_cam, build_bezier.bezier_snd_der_cam)
        self.build_bezier.getBezierTNB(self.build_bezier.bezier_pos, self.build_bezier.bezier_der,
                                       self.build_bezier.bezier_snd_der)

        ## get bezier surface mesh
        ## ref : https://mathworld.wolfram.com/Tube.html
        # build_bezier.getBezierSurface(build_bezier.bezier_pos_cam)
        self.build_bezier.getBezierSurface(self.build_bezier.bezier_pos)

        # self.build_bezier.createCylinderPrimitive()
        # build_bezier.createOpen3DVisualizer()
        # self.build_bezier.updateOpen3DVisualizer()

        ###========================================================
        ### Render Catheter Using PyTorch3D
        ###========================================================
        self.torch3d_render_catheter.updateCylinderPrimitive(self.build_bezier.updated_surface_vertices)
        self.torch3d_render_catheter.renderDeformedMesh(save_img_path)

        ###========================================================
        ### Loss : try
        ###========================================================
        # img_render = self.torch3d_render_catheter.render_catheter_img[0, ..., 3].unsqueeze(0).unsqueeze(0)
        # # loss = torch.mean((img_render - self.image_ref.unsqueeze(0).unsqueeze(0))**2)
        # loss = F.l1_loss(img_render, self.image_ref.unsqueeze(0).unsqueeze(0))
        # print(loss)
        # loss = torch.mean((img_render - self.image_ref.unsqueeze(0).unsqueeze(0))**2)

        # img_render_alpha = self.torch3d_render_catheter.render_catheter_img[0, ..., 3]
        # max_img_render_alpha = torch.max(self.torch3d_render_catheter.render_catheter_img[0, ..., 3])
        # img_render_alpha_norm = img_render_alpha / max_img_render_alpha
        # img_diff = torch.abs(img_render_alpha_norm - self.image_ref)

        # img_render_alpha = self.torch3d_render_catheter.render_catheter_img[0, ..., 3]
        # loss, img_render_binary = self.mask_loss(img_render_alpha.unsqueeze(0), self.image_ref.unsqueeze(0))
        # img_diff = torch.abs(img_render_binary - self.image_ref)

        # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        # ax = axes.ravel()
        # ax[0].imshow(self.image_ref.cpu().detach().numpy(), cmap=colormap.gray)
        # ax[0].set_title('raw thresholding')
        # ax[1].imshow(img_render_binary.cpu().detach().numpy(), cmap=colormap.gray)
        # ax[1].set_title('render binary')
        # ax[2].imshow(img_render_alpha.cpu().detach().numpy(), cmap=colormap.gray)
        # ax[2].set_title('raw render')
        # ax[3].imshow(img_diff.cpu().detach().numpy(), cmap=colormap.gray)
        # ax[3].set_title('difference')
        # plt.show()

        pdb.set_trace()

        loss = self.torch3d_render_catheter.render_catheter_img[0, ..., 0][1, 1]
        loss = torch.sum(self.torch3d_render_catheter.render_catheter_img[0, ..., 3])
        # loss = self.torch3d_render_catheter.render_cameras.get_projection_transform().get_matrix()[0, 0, 0]
        # loss = self.torch3d_render_catheter.updated_cylinder_primitive_mesh.verts_list()[0][0, 1]
        img_render_binary = None

        return loss, img_render_binary
