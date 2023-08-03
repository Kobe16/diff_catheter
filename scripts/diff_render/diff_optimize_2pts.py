import sys
from turtle import pd

sys.path.append('..')

import os
import numpy as np

# import transforms
# import bezier_interspace_transforms
sys.path.insert(1, '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts')
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

    def __init__(self, p_start, image_ref, cylinder_primitive_path, gpu_or_cpu):
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
        self.para_init = nn.Parameter(torch.from_numpy(
            np.array([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
                     dtype=np.float32)).to(gpu_or_cpu),
                                      requires_grad=True)

        ### GT values
        # self.para_init = nn.Parameter(
        #     torch.from_numpy(
        #         np.array([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896],
        #                  dtype=np.float32)).to(gpu_or_cpu))

        # # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)

    def forward(self, save_img_path):

        ###========================================================
        ### get Bezier Surface
        ###========================================================
        ## define a bezier curve
        print("para_init: ", self.para_init)
        print("p_start: ", self.p_start)
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

        img_render_alpha = self.torch3d_render_catheter.render_catheter_img[0, ..., 3]
        loss, img_render_binary = self.mask_loss(img_render_alpha.unsqueeze(0), self.image_ref.unsqueeze(0))
        img_diff = torch.abs(img_render_binary - self.image_ref)

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

        # pdb.set_trace()

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
    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/diff_render_1'
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

    # cylinder_primitive_path = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/cylinder_primitve.obj'
    cylinder_primitive_path = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/cylinder_primitve.obj'


    

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
    # save_img_path = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/torch3d_rendered_imgs/' + 'torch3d_render_' + str(
    #     img_id) + '.jpg'  # save the figure to file
    save_img_path = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/' + 'torch3d_render_' + str(
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

    # Convert image to grayscale
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_RGB2GRAY)

    # Applies binary thresholding operation to grayscale img. 
    # Sets all pixel values below the threshold value of 80 to 0. 
    # Sets all pixel values above or equal to 80 to 255
    ret, img_ref_thre = cv2.threshold(img_ref_gray.copy(), 80, 255, cv2.THRESH_BINARY)

    # Creates a binary image by replacing all pixel values equal to 255 with 1 (leaves other pixel values unchanged) 
    img_ref_binary = np.where(img_ref_thre == 255, 1, img_ref_thre)

    diff_model = DiffOptimizeModel(p_start=p_start,
                                   image_ref=img_ref_binary,
                                   cylinder_primitive_path=cylinder_primitive_path,
                                   gpu_or_cpu=gpu_or_cpu).to(gpu_or_cpu)

    # for param in diff_model.parameters():
    #     print(type(param), param.size())

    # pdb.set_trace()

    optimizer = torch.optim.Adam(diff_model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(diff_model.parameters(), lr=1e-8)


    # loss, _ = diff_model(save_img_path)
    # print(loss)

    loop = tqdm(range(100))
    for loop_id in loop:
        print('\n *********************************LOOP_ID: ', loop_id)

        # save_img_path = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/torch3d_rendered_imgs/' + 'torch3d_render_' + str(
        #     loop_id) + '.jpg'  # save the figure to file
        save_img_path = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/torch3d_rendered_imgs/' + 'torch3d_render_' + str(
            loop_id) + '.jpg'  # save the figure to file
        

        optimizer.zero_grad()
        loss = diff_model(save_img_path)

        print("Loss BEFORE BACK PROP: ", loss)
        for name, param in diff_model.named_parameters():
            print(f'Parameter: {name}, Gradient: {param.grad}')


        # pdb.set_trace()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(diff_model.parameters(), max_norm=1.0)
        optimizer.step()

        # loop.set_description('Optimizing (loss %.4f)' % loss.data)
        # for name, param in diff_model.state_dict().items():
        #     print(name, param.data().cpu.detach().numpy())
        for param in diff_model.parameters():
            print(param.data)
        print("Loss : ", loss)