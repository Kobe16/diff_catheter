import sys
# from turtle import pd

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
import matplotlib as mpl

import torch.nn as nn

import pdb

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

import torch.nn.functional as nn_func


class ConstructionBezier(nn.Module):

    def __init__(self, cam_K, cam_RT_H, gpu_or_cpu):
        super().__init__()

        self.gpu_or_cpu = gpu_or_cpu

        ## initialize camera parameters
        # self.setCameraParams(camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y,
        #                      camera_settings.image_size_x, camera_settings.image_size_y, camera_settings.extrinsics,
        #                      camera_settings.intrinsics)
        # self.cam_K = torch.tensor([[879.19277423, 0., 320.], [0., 879.19277423, 240.], [0., 0., 1.]]).to(gpu_or_cpu)

        # # camera E parameters
        # cam_RT_H = torch.tensor([[1., 0., 0., -20.5], [-0., -0., -1., 0.5], [-0., -1., -0., 64.5], [0., 0., 0.,
        #                                                                                             1.]]).to(gpu_or_cpu)
        # # cam_RT_H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]).to(gpu_or_cpu)
        # invert_y = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]).to(gpu_or_cpu)
        # self.cam_RT_H = torch.matmul(invert_y, cam_RT_H)

        self.cam_K = torch.as_tensor(cam_K).float().to(self.gpu_or_cpu)
        self.cam_RT_H = torch.as_tensor(cam_RT_H).float().to(self.gpu_or_cpu)

        ## initialize a catheter
        n_beziers = 1
        self.bezier_set = BezierSet(n_beziers)

        self.bezier_num_samples = 101
        self.bezier_surface_resolution = 40

        # self.bezier_radius = 0.0015

    # def loadRawImage(self, img_path):
    #     raw_img_rgb = cv2.imread(img_path)
    #     self.img_ownscale = 1.0
    #     self.raw_img_rgb = cv2.resize(
    #         raw_img_rgb, (int(raw_img_rgb.shape[1] / self.img_ownscale), int(raw_img_rgb.shape[0] / self.img_ownscale)))
    #     self.raw_img_gray = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

    def setCameraParams(self, fx, fy, cx, cy, size_x, size_y, camera_extrinsics, camera_intrinsics):
        """
        Set intrinsic and extrinsic camera parameters

        Args:
            fx (float): horizontal direction focal length
            fy (float): vertical direction focal length
            cx (float): horizontal center of image
            cy (float): vertical center of image
            size_x (int): width of image
            size_y (int): height of image
            camera_extrinsics ((4, 4) numpy array): RT matrix 
            camera_intrinsics ((3, 3) numpy array): K matrix 
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.size_x = size_x
        self.size_y = size_y
        # self.cam_RT_H = torch.as_tensor(camera_extrinsics)
        self.cam_K = torch.as_tensor(camera_intrinsics)

        # camera E parameters
        cam_RT_H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        invert_y = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.cam_RT_H = torch.matmul(invert_y, cam_RT_H)

    def getBezierRadius(self, r_start, r_end, radius_scale):
        # self.bezier_radius = torch.linspace(r_start, r_end, steps=self.bezier_num_samples)

        d_step = (r_end - r_start) / self.bezier_num_samples
        # self.bezier_radius = torch.arange(r_start, r_end, step)

        step_arange = d_step * torch.arange(0, self.bezier_num_samples, step=1).to(self.gpu_or_cpu)

        self.bezier_radius = (r_start + step_arange) * radius_scale

        # pdb.set_trace()

    def getQuadraticBezierCurve(self, p_mid_end, p_start=None):

        p_mid = p_mid_end[0:3]
        p_end = p_mid_end[3:6]
        p_c2 = 4 / 3 * p_mid - 1 / 3 * p_start
        p_c1 = 4 / 3 * p_mid - 1 / 3 * p_end
        # self.control_pts = torch.vstack((p_start, c2, p_end, c1))

        sample_list = torch.linspace(0, 1, self.bezier_num_samples).to(self.gpu_or_cpu)

        # Get positions and normals from samples along bezier curve
        self.bezier_pos = torch.zeros(self.bezier_num_samples, 3).to(self.gpu_or_cpu)
        self.bezier_der = torch.zeros(self.bezier_num_samples, 3).to(self.gpu_or_cpu)
        self.bezier_snd_der = torch.zeros(self.bezier_num_samples, 3).to(self.gpu_or_cpu)
        for i, s in enumerate(sample_list):
            self.bezier_pos[i, :] = (1 - s)**3 * p_start + 3 * s * (1 - s)**2 * p_c1 + 3 * (1 - s) * s**2 * p_c2 + s**3 * p_end
            # self.bezier_der[i, :] = -(1 - s)**2 * p_start + ((1 - s)**2 - 2 * s *
            #                                                  (1 - s)) * p_c1 + (-s**2 + 2 *
            #                                                                     (1 - s) * s) * p_c2 + s**2 * p_end
            # self.bezier_snd_der[i, :] = 6 * (1 - s) * (p_c2 - 2 * p_c1 + p_start) + 6 * s * (p_end - 2 * p_c2 + p_c1)
            self.bezier_der[i, :] = 3 * (1 - s)**2 * (p_c1 - p_start) + 6 * (1 - s) * s * (p_c2 - p_c1) + 3 * s**2 * (p_end - p_c2)
            self.bezier_snd_der[i, :] = 6 * (1 - s) * (p_c2 - 2 * p_c1 + p_start) + 6 * s * (p_end - 2 * p_c2 + p_c1)

        # Convert positions and normals to camera frame
        pos_bezier_H = torch.cat((self.bezier_pos, torch.ones(self.bezier_num_samples, 1).to(self.gpu_or_cpu)), dim=1)

        bezier_pos_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(pos_bezier_H, 0, 1)), 0, 1)
        # self.bezier_pos_cam = bezier_pos_cam_H[1:, :-1]  ## without including the first point
        self.bezier_pos_cam = bezier_pos_cam_H[:, :-1]

        der_bezier_H = torch.cat((self.bezier_der, torch.zeros((self.bezier_num_samples, 1)).to(self.gpu_or_cpu)), dim=1)
        bezier_der_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[0:, :], 0, 1)), 0, 1)
        self.bezier_der_cam = bezier_der_cam_H[:, :-1]

        der_bezier_H = torch.cat((self.bezier_der, torch.zeros((self.bezier_num_samples, 1)).to(self.gpu_or_cpu)), dim=1)
        bezier_der_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[0:, :], 0, 1)), 0, 1)
        self.bezier_der_cam = bezier_der_cam_H[:, :-1]

        der_snd_bezier_H = torch.cat((self.bezier_snd_der, torch.zeros((self.bezier_num_samples, 1)).to(self.gpu_or_cpu)), dim=1)
        bezier_snd_der_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_snd_bezier_H[0:, :], 0, 1)), 0, 1)
        self.bezier_snd_der_cam = bezier_snd_der_cam_H[:, :-1]

        ## get project image centerline
        self.bezier_proj_img = self.getProjPointCam(self.bezier_pos_cam[0:], self.cam_K)
        self.bezier_der_proj_img = self.getProjPointCam(self.bezier_der_cam[0:], self.cam_K)

        # pdb.set_trace()

    def getCubicBezierCurve(self, p_mid_end, p_start=None):

        # p_mid = para_gt[0:3]
        # p_end = para_gt[3:6]
        # p_c2 = 4 / 3 * p_mid - 1 / 3 * p_start
        # p_c1 = 4 / 3 * p_mid - 1 / 3 * p_end

        # need to confirm the order of c1/c2?
        p_c1 = p_mid_end[0:3]
        p_c2 = p_mid_end[3:6]
        p_end = p_mid_end[6:9]

        # pdb.set_trace()

        sample_list = torch.linspace(0, 1, self.bezier_num_samples).to(self.gpu_or_cpu)

        # Get positions and normals from samples along bezier curve
        self.bezier_pos = torch.zeros(self.bezier_num_samples, 3).to(self.gpu_or_cpu)
        self.bezier_der = torch.zeros(self.bezier_num_samples, 3).to(self.gpu_or_cpu)
        self.bezier_snd_der = torch.zeros(self.bezier_num_samples, 3).to(self.gpu_or_cpu)
        for i, s in enumerate(sample_list):
            self.bezier_pos[i, :] = (1 - s)**3 * p_start + 3 * s * (1 - s)**2 * p_c1 + 3 * (1 - s) * s**2 * p_c2 + s**3 * p_end
            self.bezier_der[i, :] = -(1 - s)**2 * p_start + ((1 - s)**2 - 2 * s * (1 - s)) * p_c1 + (-s**2 + 2 * (1 - s) * s) * p_c2 + s**2 * p_end
            self.bezier_snd_der[i, :] = 6 * (1 - s) * (p_c2 - 2 * p_c1 + p_start) + 6 * s * (p_end - 2 * p_c2 + p_c1)

        # Convert positions and normals to camera frame
        pos_bezier_H = torch.cat((self.bezier_pos, torch.ones(self.bezier_num_samples, 1).to(self.gpu_or_cpu)), dim=1)

        bezier_pos_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(pos_bezier_H, 0, 1)), 0, 1)
        # self.bezier_pos_cam = bezier_pos_cam_H[1:, :-1]  ## without including the first point
        self.bezier_pos_cam = bezier_pos_cam_H[:, :-1]

        der_bezier_H = torch.cat((self.bezier_der, torch.zeros((self.bezier_num_samples, 1)).to(self.gpu_or_cpu)), dim=1)
        bezier_der_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[0:, :], 0, 1)), 0, 1)
        self.bezier_der_cam = bezier_der_cam_H[:, :-1]

        der_bezier_H = torch.cat((self.bezier_der, torch.zeros((self.bezier_num_samples, 1)).to(self.gpu_or_cpu)), dim=1)
        bezier_der_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[0:, :], 0, 1)), 0, 1)
        self.bezier_der_cam = bezier_der_cam_H[:, :-1]

        der_snd_bezier_H = torch.cat((self.bezier_snd_der, torch.zeros((self.bezier_num_samples, 1)).to(self.gpu_or_cpu)), dim=1)
        bezier_snd_der_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_snd_bezier_H[0:, :], 0, 1)), 0, 1)
        self.bezier_snd_der_cam = bezier_snd_der_cam_H[:, :-1]

        ## get project image centerline + der
        self.bezier_proj_img = self.getProjPointCam(self.bezier_pos_cam[0:], self.cam_K)
        self.bezier_der_proj_img = self.getProjPointCam(self.bezier_der_cam[0:], self.cam_K)

        # pdb.set_trace()

    ## get the ground truth skeleton projected in the image
    def getGroundTruthCenterlineCam(self, gt_centline_3d):
        self.gt_centline_3d = torch.from_numpy(gt_centline_3d.astype(np.float32)).to(self.gpu_or_cpu)
        # self.gt_centline_3d = gt_centline_3d

        num_gt_centline_3d = self.gt_centline_3d.shape[0]

        # Convert to camera frame
        gt_centline_3d_H = torch.cat((self.gt_centline_3d, torch.ones(num_gt_centline_3d, 1).to(self.gpu_or_cpu)), dim=1)

        gt_centline_3d_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(gt_centline_3d_H, 0, 1)), 0, 1)
        self.gt_centline_3d_cam = gt_centline_3d_cam_H[:, :-1]

        # Convert to image
        self.gt_centline_proj_img = self.getProjPointCam(self.gt_centline_3d_cam[:], self.cam_K)

        # return self.gt_centline_proj_img

    def getProjPointCam(self, p, cam_K):
        # p is of size R^(Nx3)
        if p.shape == (3, ):
            p = torch.unsqueeze(p, dim=0)

        divide_z = torch.div(torch.transpose(p[:, :-1], 0, 1), p[:, -1])
        divide_z = torch.cat((divide_z, torch.ones(1, p.shape[0]).to(self.gpu_or_cpu)), dim=0).float()

        return torch.transpose(torch.matmul(cam_K, divide_z)[:-1, :], 0, 1)

    def getBezierTNB(self, bezier_pos, bezier_der, bezier_snd_der):

        bezier_der_n = torch.linalg.norm(bezier_der, ord=2, dim=1)
        self.bezier_tangent = bezier_der / torch.unsqueeze(bezier_der_n, dim=1)

        # self.bezier_nml = self.bezier_tangent / torch.unsqueeze(bezier_nml_numerator_n, dim=1)

        bezier_nml_numerator = torch.linalg.cross(bezier_der, torch.linalg.cross(bezier_snd_der, bezier_der))
        vanish_n_nml = torch.mul(bezier_der_n, torch.linalg.norm(torch.linalg.cross(bezier_snd_der, bezier_der), ord=2, dim=1))

        bezier_bnml_numerator = torch.linalg.cross(bezier_der, bezier_snd_der)
        vanish_n_bnml = torch.linalg.norm(bezier_bnml_numerator, ord=2, dim=1)

        # # ### following is going to find the vanishing case : normal
        # single_diff_n_nml = torch.diff(vanish_n_nml, dim=0)
        # double_diff_n_nml = torch.diff(single_diff_n_nml, dim=0)

        # ### thresholding :  +1 is offset because of double diff
        # # thres_nml = double_diff_n_nml.mean() * 10
        # thres_nml = 0.1
        # double_thres_id_nml = torch.nonzero(nn_func.relu(double_diff_n_nml - thres_nml)) + 2
        # if double_thres_id_nml.shape[0] > 0:
        #     zero_build_nml = (double_thres_id_nml[0] * 0).unsqueeze(0)
        #     candi_id_nml = torch.cat(((zero_build_nml, double_thres_id_nml, zero_build_nml + vanish_n_nml.shape[0] - 1)))
        #     candi_id_diff_nml = torch.diff(candi_id_nml, dim=0)

        #     final_candi_nml = torch.clone(zero_build_nml)
        #     sum_noncandi_nml = zero_build_nml[0]
        #     for i in range(candi_id_diff_nml.shape[0]):
        #         if candi_id_diff_nml[i] > 2:
        #             final_candi_nml = torch.cat((final_candi_nml, (candi_id_diff_nml[i] + sum_noncandi_nml + final_candi_nml[-1]).unsqueeze(0)))
        #         else:
        #             sum_noncandi_nml = sum_noncandi_nml + candi_id_diff_nml[i]

        #     if sum_noncandi_nml[0] == 0:
        #         off_set_nml = 1
        #     else:
        #         off_set_nml = 0

        #     # adding slices
        #     nonvanish_n_nml = vanish_n_nml[0:final_candi_nml[1]]
        #     for i in range(final_candi_nml.shape[0]):
        #         if i == 0:
        #             continue
        #         if i == final_candi_nml.shape[0] - 1:
        #             continue

        #         if (i % 2) == 0:
        #             ## even index
        #             nonvanish_n_nml = torch.cat((nonvanish_n_nml, vanish_n_nml[final_candi_nml[i] - off_set_nml:final_candi_nml[i + 1] + 1]))
        #         else:
        #             ## odd index : need to -1, in order to
        #             nonvanish_n_nml = torch.cat((nonvanish_n_nml, -vanish_n_nml[final_candi_nml[i]:final_candi_nml[i + 1] - off_set_nml]))
        # else:
        #     nonvanish_n_nml = vanish_n_nml

        # # ### following is going to find the vanishing case : binormal
        # single_diff_n_bnml = torch.diff(vanish_n_bnml, dim=0)
        # double_diff_n_bnml = torch.diff(single_diff_n_bnml, dim=0)

        # ### thresholding :  +1 is offset because of double diff
        # # thres_bnml = double_diff_n_bnml.mean() * 10
        # thres_bnml = 0.01
        # double_thres_id_bnml = torch.nonzero(nn_func.relu(double_diff_n_bnml - thres_bnml)) + 2
        # if double_thres_id_bnml.shape[0] > 0:
        #     zero_build_bnml = (double_thres_id_bnml[0] * 0).unsqueeze(0)
        #     candi_id_bnml = torch.cat(((zero_build_bnml, double_thres_id_bnml, zero_build_bnml + vanish_n_nml.shape[0] - 1)))
        #     candi_id_diff_bnml = torch.diff(candi_id_bnml, dim=0)

        #     final_candi_bnml = torch.clone(zero_build_bnml)
        #     sum_noncandi_bnml = zero_build_bnml[0]
        #     for i in range(candi_id_diff_bnml.shape[0]):
        #         if candi_id_diff_bnml[i] > 2:
        #             final_candi_bnml = torch.cat((final_candi_bnml, (candi_id_diff_bnml[i] + sum_noncandi_bnml + final_candi_bnml[-1]).unsqueeze(0)))
        #         else:
        #             sum_noncandi_bnml = sum_noncandi_bnml + candi_id_diff_bnml[i]

        #     if sum_noncandi_bnml[0] == 0:
        #         off_set_bnml = 1
        #     else:
        #         off_set_bnml = 0

        #     # adding slices
        #     nonvanish_n_bnml = vanish_n_bnml[0:final_candi_bnml[1]]
        #     for i in range(final_candi_bnml.shape[0]):
        #         if i == 0:
        #             continue
        #         if i == final_candi_bnml.shape[0] - 1:
        #             continue

        #         if (i % 2) == 0:
        #             ## even index
        #             nonvanish_n_bnml = torch.cat((nonvanish_n_bnml, vanish_n_bnml[final_candi_bnml[i] - off_set_bnml:final_candi_bnml[i + 1] + 1]))
        #         else:
        #             ## odd index : need to -1, in order to
        #             nonvanish_n_bnml = torch.cat((nonvanish_n_bnml, -vanish_n_bnml[final_candi_bnml[i]:final_candi_bnml[i + 1] - off_set_bnml]))
        # else:
        #     nonvanish_n_bnml = vanish_n_bnml

        # pdb.set_trace()

        self.vanish_bezier_nml = bezier_nml_numerator / vanish_n_nml.unsqueeze(1)
        self.vanish_bezier_bnml = bezier_bnml_numerator / vanish_n_bnml.unsqueeze(1)

        # self.nonvanish_bezier_nml = bezier_nml_numerator / nonvanish_n_nml.unsqueeze(1)
        # self.nonvanish_bezier_bnml = bezier_bnml_numerator / nonvanish_n_bnml.unsqueeze(1)

        # assert not torch.any(torch.isnan(self.nonvanish_bezier_nml))
        # assert not torch.any(torch.isnan(self.nonvanish_bezier_bnml))
        assert not torch.any(torch.isnan(self.vanish_bezier_nml))
        assert not torch.any(torch.isnan(self.vanish_bezier_bnml))

        # fig, axes = plt.subplots(6, 1, figsize=(9, 9))
        # ax = axes.ravel()

        # nonvanish_bezier_nml_npy = self.nonvanish_bezier_nml.cpu().detach().numpy()
        # nonvanish_bezier_bnml_npy = self.nonvanish_bezier_bnml.cpu().detach().numpy()
        # vanish_bezier_nml_npy = self.vanish_bezier_nml.cpu().detach().numpy()
        # vanish_bezier_bnml_npy = self.vanish_bezier_bnml.cpu().detach().numpy()

        # # ax[0].plot(single_diff_n_nml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[1].plot(double_diff_n_nml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[2].plot(vanish_n_nml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[3].plot(nonvanish_n_nml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)

        # # ax[0].plot(nonvanish_bezier_nml_npy[:, 0], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[1].plot(nonvanish_bezier_nml_npy[:, 1], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[2].plot(nonvanish_bezier_nml_npy[:, 2], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[3].plot(vanish_bezier_nml_npy[:, 0], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[4].plot(vanish_bezier_nml_npy[:, 1], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[5].plot(vanish_bezier_nml_npy[:, 2], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)

        # # ax[0].plot(single_diff_n_bnml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[1].plot(double_diff_n_bnml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[2].plot(vanish_n_bnml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[3].plot(nonvanish_n_bnml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)

        # # ax[0].plot(nonvanish_bezier_bnml_npy[:, 0], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[1].plot(nonvanish_bezier_bnml_npy[:, 1], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[2].plot(nonvanish_bezier_bnml_npy[:, 2], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[3].plot(vanish_bezier_bnml_npy[:, 0], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[4].plot(vanish_bezier_bnml_npy[:, 1], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[5].plot(vanish_bezier_bnml_npy[:, 2], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)

        # ax[0].plot(vanish_n_bnml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # ax[1].plot(nonvanish_n_bnml.cpu().detach().numpy(), linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # # ax[2].plot(, linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # ax[3].plot(nonvanish_bezier_bnml_npy[:, 0], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # ax[4].plot(nonvanish_bezier_bnml_npy[:, 1], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)
        # ax[5].plot(nonvanish_bezier_bnml_npy[:, 2], linestyle='-', marker='o', color='#5F6F94', markersize=1, linewidth=1)

        # plt.show()
        # plt.tight_layout()
        # plt.close(fig)
        # pdb.set_trace()

    def getBezierSurface(self, bezier_pos):

        self.bezier_surface = torch.zeros(self.bezier_num_samples, self.bezier_surface_resolution, 3).to(self.gpu_or_cpu)

        theta_list = torch.linspace(0.0, 2 * np.pi, self.bezier_surface_resolution).to(self.gpu_or_cpu)

        for i in range(self.bezier_num_samples):
            # surface_vec = self.bezier_radius[i] * (-torch.mul(self.nonvanish_bezier_nml[i, :], torch.unsqueeze(torch.cos(theta_list), dim=1)) +
            #                                        torch.mul(self.nonvanish_bezier_bnml[i, :], torch.unsqueeze(torch.sin(theta_list), dim=1)))
            surface_vec = self.bezier_radius[i] * (-torch.mul(self.vanish_bezier_nml[i, :], torch.unsqueeze(torch.cos(theta_list), dim=1)) +
                                                   torch.mul(self.vanish_bezier_bnml[i, :], torch.unsqueeze(torch.sin(theta_list), dim=1)))

            # self.bezier_surface[i, :, :] = self.bezier_pos[i, :] + surface_vec
            self.bezier_surface[i, :, :] = bezier_pos[i, :] + surface_vec

        ### Combine the surface with "top center" + "bottom center" points
        surface_vertices = torch.reshape(self.bezier_surface, (-1, 3))
        top_center_vertice = torch.unsqueeze(bezier_pos[0, :], dim=0)
        bot_center_vertice = torch.unsqueeze(bezier_pos[-1, :], dim=0)
        self.updated_surface_vertices = torch.cat((top_center_vertice, bot_center_vertice, surface_vertices), dim=0)

    def createCylinderPrimitive(self):
        self.mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=10.0, resolution=self.bezier_surface_resolution, split=self.bezier_num_samples - 1, create_uv_map=True)
        self.mesh_cylinder.compute_vertex_normals()
        self.mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    def createOpen3DVisualizer(self):

        # Visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.register_key_callback(88, self.closeOpen3DVisualizer)
        self.vis.get_view_control().set_zoom(1.5)

        # o3d.visualization.draw_geometries([mesh_cylinder, mesh_frame])
        self.vis.add_geometry(self.mesh_cylinder)
        self.vis.add_geometry(self.mesh_frame)

    def updateOpen3DVisualizer(self):

        # self.mesh_cylinder.vertices = o3d.utility.Vector3dVector(self.updated_surface_vertices.detach().numpy())
        # self.mesh_cylinder.vertices = o3d.utility.Vector3dVector(self.updated_surface_vertices.cpu().detach().numpy())
        # print(np.asarray(self.mesh_cylinder.vertices))

        o3d.visualization.draw_geometries([self.mesh_cylinder])
        # o3d.io.write_triangle_mesh("./blender_imgs/diff_render_1.obj", self.mesh_cylinder, write_triangle_uvs=True)
        o3d.io.write_triangle_mesh("./shape_primitive/cylinder_primitve_101_40.obj", self.mesh_cylinder, write_triangle_uvs=True)

        self.vis.update_geometry(self.mesh_cylinder)
        self.vis.update_renderer()

        self.vis_view = o3d.visualization.ViewControl
        self.vis_view.camera_local_translate(0, 0, 0)

    def closeOpen3DVisualizer(vis):
        print('Closing visualizer!')

    def draw2DCenterlineImage(self, image_ref, img_render_diffable, img_ref_skeleton=None):

        ## numpy copy
        image_ref_binary = image_ref.cpu().detach().numpy().copy()
        img_render_centerline = img_render_diffable.cpu().detach().numpy().copy()

        # tangent_draw_img_rgb = centerline_draw_img_rgb.copy()
        # raw_skeleton_img_rgb = centerline_draw_img_rgb.copy()

        ## torch clone
        bezier_proj_img = torch.clone(self.bezier_proj_img)

        # Draw projected centerline
        for i in range(bezier_proj_img.shape[0] - 1):
            p1 = (int(bezier_proj_img[i, 0]), int(bezier_proj_img[i, 1]))
            p2 = (int(bezier_proj_img[i + 1, 0]), int(bezier_proj_img[i + 1, 1]))
            cv2.line(img_render_centerline, p1, p2, (0, 100, 255), 2)
            # cv2.line(raw_skeleton_img_rgb, p1, p2, (0, 100, 255), 1)

        # # Draw tangent lines every few to check they are correct
        # show_every_so_many_samples = 10
        # l = 0.1
        # for i, p in enumerate(bezier_proj_img):
        #     if i % show_every_so_many_samples != 0:
        #         continue

        #     # if not self.isPointInImage(p, tangent_draw_img_rgb.shape[1], tangent_draw_img_rgb.shape[0]):
        #     #     continue

        #     p_d = self.getProjPointCam(
        #         self.bezier_pos_cam[i] + l * self.bezier_der_cam[i] / torch.linalg.norm(self.bezier_der_cam[i]),
        #         self.cam_K)[0]

        #     # if not self.isPointInImage(p_d, tangent_draw_img_rgb.shape[1], tangent_draw_img_rgb.shape[0]):
        #     #     continue

        #     # print('Out')
        #     cv2.line(tangent_draw_img_rgb, (int(p[0]), int(p[1])), (int(p_d[0]), int(p_d[1])), (0.0, 0.0, 255.0), 1)

        # Draw raw skeletonization
        for i in range(img_ref_skeleton.shape[0] - 1):

            p1 = (int(img_ref_skeleton[i, 0]), int(img_ref_skeleton[i, 1]))
            p2 = (int(img_ref_skeleton[i + 1, 0]), int(img_ref_skeleton[i + 1, 1]))
            cv2.line(image_ref_binary, p1, p2, (0, 100, 100), 2)

        # # ---------------
        # # plot with
        # # ---------------
        # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # ax = axes.ravel()

        # ax[0].imshow(cv2.cvtColor(img_render_centerline, cv2.COLOR_BGR2RGB))
        # ax[0].set_title('2d rendered centerline')

        # ax[1].imshow(cv2.cvtColor(image_ref_binary, cv2.COLOR_BGR2RGB))
        # ax[1].set_title('2d ref skeletonization')

        # # ax[1].imshow(cv2.cvtColor(tangent_draw_img_rgb, cv2.COLOR_BGR2RGB))
        # # ax[1].set_title('2d tangents')

        # # ax[2].imshow(cv2.cvtColor(cylinder_draw_img_rgb, cv2.COLOR_BGR2RGB))
        # # ax[2].set_title('Projected cylinders')

        # plt.tight_layout()
        # plt.show()

        # cv2.imwrite('./gradient_steps_imgs/centerline_draw_img_rgb_' + str(self.GD_Iteration) + '.jpg', centerline_draw_img_rgb)
        # cv2.imwrite('./gradient_steps_imgs/tangent_draw_img_rgb_' + str(self.GD_Iteration) + '.jpg', tangent_draw_img_rgb)

        return img_render_centerline

    def draw2DKeyPointsImage(self, image_ref_rgb, img_render_diffable, img_ref_skeleton, pt_intesection_endpoints_ref, pt_intesection_endpoints_render):

        ## numpy copy
        image_ref_keypoints2d = cv2.cvtColor(cv2.cvtColor(image_ref_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        img_render_keypoints2d = img_render_diffable.cpu().detach().numpy().copy()

        bezier_proj_img = torch.clone(self.bezier_proj_img)

        p_start_ref = (int(img_ref_skeleton[0, 0]), int(img_ref_skeleton[0, 1]))
        p_end_ref = (int(img_ref_skeleton[-1, 0]), int(img_ref_skeleton[-1, 1]))
        p_intesection_ref = (int(pt_intesection_endpoints_ref[0]), int(pt_intesection_endpoints_ref[1]))

        color_red = (114, 114, 236)
        cv2.circle(image_ref_keypoints2d, p_start_ref, radius=6, color=color_red, thickness=-1)
        cv2.circle(image_ref_keypoints2d, p_end_ref, radius=6, color=color_red, thickness=-1)
        cv2.circle(image_ref_keypoints2d, p_intesection_ref, radius=6, color=color_red, thickness=-1)
        cv2.line(image_ref_keypoints2d, p_start_ref, p_intesection_ref, color_red, 1)
        cv2.line(image_ref_keypoints2d, p_end_ref, p_intesection_ref, color_red, 1)

        p_start_render = (int(bezier_proj_img[0, 0]), int(bezier_proj_img[0, 1]))
        p_end_render = (int(bezier_proj_img[-1, 0]), int(bezier_proj_img[-1, 1]))
        p_intesection_render = (int(pt_intesection_endpoints_render[0]), int(pt_intesection_endpoints_render[1]))

        # color_red = (255, 255, 255)
        # cv2.circle(img_render_keypoints2d, p_start_render, radius=6, color=color_red, thickness=-1)
        # cv2.circle(img_render_keypoints2d, p_end_render, radius=6, color=color_red, thickness=-1)
        # cv2.circle(img_render_keypoints2d, p_intesection_render, radius=6, color=color_red, thickness=-1)
        # cv2.line(img_render_keypoints2d, p_start_render, p_intesection_render, color_red, 1)
        # cv2.line(img_render_keypoints2d, p_end_render, p_intesection_render, color_red, 1)

        # # colormap = mpl.cm.binary
        # # colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        # fig = plt.figure()
        # ax = fig.gca()
        # ax.imshow(cv2.cvtColor(image_ref_keypoints2d, cv2.COLOR_BGR2RGB))
        # # ax[0].plot(centerline[:, 0].detach().numpy(),
        # #            centerline[:, 1].detach().numpy(),
        # #            'bo-',
        # #            markersize=2,
        # #            linewidth=1)
        # fig.canvas.draw()

        # # Now we can save it to a numpy array.
        # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

        # ---------------
        # plot with
        # ---------------
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        colormap = mpl.cm.binary
        colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax = axes.ravel()
        ax[0].imshow(img_render_keypoints2d, cmap=colormap, norm=colormap_norm)
        ax[0].set_title('2d rendered keypoints')
        ax[1].imshow(cv2.cvtColor(image_ref_keypoints2d, cv2.COLOR_BGR2RGB))
        ax[1].set_title('2d reference keypoints')

        # ax[1].imshow(cv2.cvtColor(tangent_draw_img_rgb, cv2.COLOR_BGR2RGB))
        # ax[1].set_title('2d tangents')

        # ax[2].imshow(cv2.cvtColor(cylinder_draw_img_rgb, cv2.COLOR_BGR2RGB))
        # ax[2].set_title('Projected cylinders')

        plt.tight_layout()
        plt.show()

        pdb.set_trace()

        return img_render_keypoints2d, image_ref_keypoints2d

    def forward(self):
        raise NotImplementedError