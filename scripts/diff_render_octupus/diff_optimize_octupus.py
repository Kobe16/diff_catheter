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

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap

from tqdm.notebook import tqdm


class DoDiffOptimization(nn.Module):

    def __init__(self, para_init, para_gt, diff_model, total_itr_steps, if_print_log=1):
        super(DoDiffOptimization, self).__init__()

        self.para_init = para_init
        self.if_print_log = if_print_log
        self.diff_model = diff_model
        self.total_itr_steps = total_itr_steps

        self.optimizer = torch.optim.Adam([self.para_init], lr=1e-3)

        self.saved_para_history = np.zeros((1, self.para_init.shape[0] + 1))
        self.GD_Iteration = 0
        self.loss = None
        self.img_render_binary = None

        self.para_gt = para_gt

    def doOptimization(self):

        def closure():
            # with autograd.detect_anomaly():
            self.optimizer.zero_grad()
            self.loss, self.img_render_binary = self.diff_model()
            self.loss.backward()

            return self.loss

        self.optimizer.zero_grad()
        loss_history = []
        last_loss = 999.0  # current loss value

        converge = False  # converge or not
        self.id_iteration = 0  # number of updates

        # while not converge and self.id_iteration < self.total_itr_steps:
        while self.id_iteration < self.total_itr_steps:

            self.optimizer.step(closure)

            # if (abs(self.loss - last_loss) < 1e-6):
            #     converge = True

            self.id_iteration += 1

            if self.if_print_log:
                print("Curr grad : ", self.para_init.grad)
                print("Curr para : ", self.para_init)
                print("Curr loss : ", self.loss)
                print("---------------- FINISH ", self.id_iteration, " ^_^ ITER ---------------- \n")

            last_loss = torch.clone(self.loss)
            loss_history.append(last_loss)

            saved_value = np.hstack((last_loss.cpu().detach().numpy(), self.para_init.cpu().detach().numpy()))
            self.saved_para_history = np.vstack((self.saved_para_history, saved_value))

            save_img_path = '/home/fei/diff_catheter/scripts/diff_render/torch3d_rendered_imgs/' + 'render_' + str(
                self.id_iteration) + '.jpg'  # save the figure to file
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            ax = axes.ravel()
            ax[0].imshow(self.diff_model.image_ref.cpu().detach().numpy(), cmap=colormap.gray)
            ax[0].set_title('raw thresholding')
            ax[1].imshow(self.img_render_binary.cpu().detach().numpy(), cmap=colormap.gray)
            ax[1].set_title('render binary')
            # ax[2].imshow(img_render_alpha.cpu().detach().numpy(), cmap=colormap.gray)
            # ax[2].set_title('raw render')
            # ax[3].imshow(img_diff.cpu().detach().numpy(), cmap=colormap.gray)
            # ax[3].set_title('difference')
            # plt.show()
            fig.savefig(save_img_path)
            plt.close(fig)

            save_mesh_path = '/home/fei/diff_catheter/scripts/diff_render/torch3d_rendered_imgs/meshes' + 'mesh_' + str(
                self.id_iteration) + '.obj'  # save the figure to file
            self.diff_model.saveUpdatedMesh(save_mesh_path)

            # pdb.set_trace()

        print("Final --->", self.para_init.cpu().detach())
        print("GT    --->", self.para_gt)
        print("Error --->", torch.abs(self.para_init.cpu().detach() - self.para_gt))
        # np.savetxt(self.save_dir + '/final_optimized_para.csv', self.saved_opt_history, delimiter=",")

        # # plt.plot(loss_history)
        # plt.plot(loss_history, marker='o', linestyle='-', linewidth=1, markersize=4)
        # plt.show()

        # return self.saved_opt_history, self.para


if __name__ == '__main__':

    # p_start = torch.tensor([0.5, 0.5, 0.5])

    # para_gt = torch.tensor([
    #     0.5, 0.5, 0.5, 35.37860277150629, 3.7807340989748486, -2.7807340989748557, 36.085861528232165, 7.06569109504143,
    #     -6.065691095041432, 1.9999573999981042, 0.40002940478525545
    # ],
    #                        dtype=torch.float)

    # para_init = torch.tensor([
    #     0.5, 0.5, 0.5, 35.37860277150629, 3.7820790965179647, -2.78207909651797, 36.085861528232165, 7.06569109504143,
    #     -6.065691095041432, 1.9999573999981042, 0.40002940478525545
    # ],
    #                          dtype=torch.float)

    case_naming = '/home/fei/diff_catheter/data/octupus_frames/top/frame_0199'
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

    cylinder_primitive_path = '/home/fei/diff_catheter/scripts/diff_render_octupus/shape_primitive/cylinder_primitve.obj'

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

    para_gt = torch.tensor([
        0.5, 0.5, 0.5, 35.37860277150629, 3.7807340989748486, -2.7807340989748557, 36.085861528232165, 7.06569109504143,
        -6.065691095041432, 1.9999573999981042, 0.40002940478525545
    ],
                           dtype=torch.float)

    para_init = torch.tensor([
        0.5, 0.5, 0.5, 35.37860277150629, 3.7820790965179647, -2.78207909651797, 36.085861528232165, 7.06569109504143,
        -6.065691095041432, 1.9999573999981042, 0.40002940478525545
    ],
                             dtype=torch.float).to(gpu_or_cpu)

    para_init.requires_grad = True

    total_itr_steps = 100

    img_ref_rgb = cv2.imread(img_save_path)
    # img_ref_rgb = cv2.resize(img_ref_rgb, (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_RGB2GRAY)
    ret, img_ref_thre = cv2.threshold(img_ref_gray.copy(), 245, 255, cv2.THRESH_BINARY_INV)
    img_ref_thre_inv = cv2.bitwise_not(img_ref_thre)

    img_ref_binary = np.where(img_ref_thre_inv == 255, 1, img_ref_thre_inv)

    diff_model = DiffOptimizeModel(para_init=para_init[3:9],
                                   p_start=para_init[0:3],
                                   radius_start=para_init[9],
                                   radius_end=para_init[10],
                                   image_ref=img_ref_binary,
                                   cylinder_primitive_path=cylinder_primitive_path,
                                   gpu_or_cpu=gpu_or_cpu).to(gpu_or_cpu)

    do_diff = DoDiffOptimization(para_init=para_init,
                                 para_gt=para_gt,
                                 diff_model=diff_model,
                                 total_itr_steps=total_itr_steps,
                                 if_print_log=1)

    do_diff.doOptimization()