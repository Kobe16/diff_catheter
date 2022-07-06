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

import pytorch3d

import torch.nn as nn
import matplotlib.cm as colormap

from tqdm.notebook import tqdm


class DiffOptimizeModel(nn.Module):

    def __init__(self, p_start, image_ref, cylinder_primitive_path, gpu_or_cpu):
        super().__init__()

        self.build_bezier = ConstructionBezier()
        # self.build_bezier.to(gpu_or_cpu)

        self.torch3d_render_catheter = DiffRenderCatheter(self.build_bezier.cam_RT_H, self.build_bezier.cam_K,
                                                          gpu_or_cpu)
        # self.torch3d_render_catheter.to(gpu_or_cpu)
        self.torch3d_render_catheter.loadCylinderPrimitive(cylinder_primitive_path)

        self.p_start = p_start.to(gpu_or_cpu)

        # para_init = torch.tensor([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
        #  dtype=torch.float)
        self.para_init = nn.Parameter(
            torch.from_numpy(
                np.array([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
                         dtype=np.float32)).to(gpu_or_cpu))

        # # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)

    def forward(self, save_img_path):

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
        ###========================================================\
        self.torch3d_render_catheter.updateCylinderPrimitive(self.build_bezier.updated_surface_vertices)
        self.torch3d_render_catheter.renderDeformedMesh(save_img_path)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        # ax = axes.ravel()
        # ax[0].imshow(self.image_ref.cpu().detach().numpy(), cmap=colormap.gray)
        # ax[0].set_title('raw thresholding')
        # ax[1].imshow(self.torch3d_render_catheter.render_catheter_img[0, ..., 3].cpu().detach().numpy(),
        #              cmap=colormap.gray)
        # ax[1].set_title('render')
        # plt.show()

        # pdb.set_trace()

        loss = torch.sum((self.torch3d_render_catheter.render_catheter_img[0, ..., 3] - self.image_ref)**2)

        return loss


if __name__ == '__main__':

    # para_gt = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896], dtype=torch.float)
    p_start = torch.tensor([0.02, 0.002, 0.0])

    # para_init = torch.tensor([0.02, 0.002, 0.15, 0.03, -0.05, 0.2], dtype=torch.float, requires_grad=True)
    para_init = torch.tensor([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
                             dtype=torch.float)
    # para_init = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896],
    #                          dtype=torch.float)

    # case_naming = '/home/fei/ARCLab-CCCatheter/scripts/diff_render/blender_imgs/diff_render_1'
    case_naming = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/diff_render_1'
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

    cylinder_primitive_path = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/cylinder_primitve.obj'

    ###========================================================
    ### Render catheter using Blender
    ###========================================================
    # raw_render_catheter = BlenderRenderCatheter()
    # raw_render_catheter.set_bezier_in_blender(para_gt.detach().numpy(), p_start.detach().numpy())

    # raw_render_catheter.render_bezier_in_blender(cc_specs_path, img_save_path, target_specs_path, viewpoint_mode,
    #                                           transparent_mode)

    ###========================================================
    ### Build Bezier Suface Mesh
    ###========================================================
    # build_bezier = ConstructionBezier()

    # ## define a bezier curve
    # build_bezier.getBezierCurve(para_init, p_start)

    # ## get the bezier in TNB frame, in order to build a tube mesh
    # # build_bezier.getBezierTNB(build_bezier.bezier_pos_cam, build_bezier.bezier_der_cam,
    # #                            build_bezier.bezier_snd_der_cam)
    # build_bezier.getBezierTNB(build_bezier.bezier_pos, build_bezier.bezier_der, build_bezier.bezier_snd_der)

    # ## get bezier surface mesh
    # ## ref : https://mathworld.wolfram.com/Tube.html
    # # build_bezier.getBezierSurface(build_bezier.bezier_pos_cam)
    # build_bezier.getBezierSurface(build_bezier.bezier_pos)

    # build_bezier.createCylinderPrimitive()
    # # build_bezier.createOpen3DVisualizer()
    # build_bezier.updateOpen3DVisualizer()

    # ## load the raw RGB image
    # # build_bezier.loadRawImage(img_save_path)
    # build_bezier.proj_bezier_img = build_bezier.getProjPointCam(build_bezier.bezier_pos_cam, build_bezier.cam_K)
    # # build_bezier.draw2DCenterlineImage()

    ###========================================================
    ### Differentiable Rendering
    ###========================================================
    # torch3d_render_catheter = DiffRenderCatheter(build_bezier.cam_RT_H, build_bezier.cam_K)
    # torch3d_render_catheter.loadCylinderPrimitive(cylinder_primitive_path)

    # torch3d_render_catheter.updateCylinderPrimitive(build_bezier.updated_surface_vertices)

    img_id = 0
    save_img_path = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/torch3d_rendered_imgs/' + 'torch3d_render_' + str(
        img_id) + '.jpg'  # save the figure to file
    # torch3d_render_catheter.renderDeformedMesh(save_img_path)

    ###========================================================
    ### Optimization Rendering
    ###========================================================
    ## Set the cuda device
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device("cuda:0")
        torch.cuda.set_device(gpu_or_cpu)
    else:
        gpu_or_cpu = torch.device("cpu")

    img_ref_rgb = cv2.imread(img_save_path)
    # img_ref_rgb = cv2.resize(img_ref_rgb, (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_RGB2GRAY)
    ret, img_ref_binary = cv2.threshold(img_ref_gray.copy(), 80, 255, cv2.THRESH_BINARY)

    diff_model = DiffOptimizeModel(p_start=p_start,
                                   image_ref=img_ref_binary,
                                   cylinder_primitive_path=cylinder_primitive_path,
                                   gpu_or_cpu=gpu_or_cpu).to(gpu_or_cpu)

    # for param in diff_model.parameters():
    #     print(type(param), param.size())

    # pdb.set_trace()

    optimizer = torch.optim.Adam(diff_model.parameters(), lr=1e-3)

    # loss, _ = diff_model(save_img_path)
    # print(loss)

    loop = tqdm(range(100))
    for loop_id in loop:

        save_img_path = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/torch3d_rendered_imgs/' + 'torch3d_render_' + str(
            loop_id) + '.jpg'  # save the figure to file

        optimizer.zero_grad()
        loss = diff_model(save_img_path)

        pdb.set_trace()
        loss.backward()
        optimizer.step()

        # loop.set_description('Optimizing (loss %.4f)' % loss.data)
        # for name, param in diff_model.state_dict().items():
        #     print(name, param.data().cpu.detach().numpy())
        for param in diff_model.parameters():
            print(param.data)
        print("Loss : ", loss)