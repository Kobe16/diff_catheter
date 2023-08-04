import torch

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap
import matplotlib.pyplot as plt

import skimage.morphology as skimage_morphology
import cv2

import pdb

import numpy as np


class ContourLoss(nn.Module):

    def __init__(self, device):
        super(ContourLoss, self).__init__()
        self.device = device

    def forward(self, img_render, img_ref):
        # Binarize img_render [0.0, 1.0] -> {0., 1.}
        # img_render = (img_render >= 0.1).float()   # Thresholding is NOT differentiable
        img_render = 1 / (1 + torch.exp(-100 * (img_render - 0.1)))  # Differentiable binarization (approximation)
        mask = (img_render < 0.1)
        img_render = img_render * mask  # Zero out values above the threshold 0.5

        # Convert from (B x H x W) to (B x C x H x W)
        img_render = torch.unsqueeze(img_render, 1)

        # Apply Laplacian operator to grayscale images to find contours
        kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]).to(self.device)

        contours = F.conv2d(img_render, kernel, padding=1)
        contours = torch.clamp(contours, min=0, max=255)

        # Convert from (B x C x H x W) back to (B x H x W)
        contours = torch.squeeze(contours, 1)

        # Compute the Chamfer distance between two images
        # Selecting indices is NOT differentiable -> use tanh(x) or 2 / (1 + e^(-100(x))) - 1 for differentiable thresholding
        # -> apply element-wise product between contours and distance maps
        contours = torch.tanh(contours)
        diff_dist = contours * img_ref  # element-wise product

        dist = diff_dist.sum() / contours.shape[0]
        assert (dist >= 0)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.ravel()
        ax[0].imshow(img_ref.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        ax[0].set_title('raw thresholding')
        ax[1].imshow(img_render.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        ax[1].set_title('render thresholding')
        ax[2].imshow(contours.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        ax[2].set_title('render thresholding')
        ax[3].imshow(diff_dist.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        ax[3].set_title('difference')
        plt.show()

        pdb.set_trace()

        return dist


class MaskLoss(nn.Module):

    def __init__(self, device):
        super(MaskLoss, self).__init__()
        self.device = device
        # self.mse_loss = nn.MSELoss()

    def forward(self, img_render, img_ref):
        # Binarize img_render [0.0, 1.0] -> {0., 1.}
        # img_render = (img_render >= 0.1).float()   # Thresholding is NOT differentiable
        
        # img_render = 1 / (1 + torch.exp(-100 * (img_render - 0.1)))  # Differentiable binarization (approximation)
        # mask = (img_render > 0.1)
        # img_render = img_render * mask  # Zero out values above the threshold 0.5

        img_render_binary = img_render.squeeze()

        # img_diff = torch.abs(img_render - img_ref)**2

        dist = torch.sum((img_render -img_ref) ** 2)
        # dist = self.mse_loss(img_render, img_ref)
        assert (dist >= 0)

        # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        # ax = axes.ravel()
        # ax[0].imshow(img_ref.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        # ax[0].set_title('raw thresholding')
        # ax[1].imshow(img_render.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        # ax[1].set_title('render thresholding')
        # ax[2].imshow(img_diff.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        # ax[2].set_title('render thresholding')
        # plt.show()

        # pdb.set_trace()

        return dist, img_render_binary


class CenterlineLoss(nn.Module):

    def __init__(self, device):
        super(CenterlineLoss, self).__init__()
        self.device = device

        self.img_raw_skeleton = None

    def forward(self, bezier_proj_img, img_ref):

        self.get_raw_centerline(img_ref)
        
        loss_centerline = (bezier_proj_img[-1, 0] - self.img_raw_skeleton[0, 1])**2 + (bezier_proj_img[-1, 1] - self.img_raw_skeleton[0, 0])**2

        # pdb.set_trace()

        return loss_centerline

    def get_raw_centerline(self, img_ref):

        img_ref = img_ref.cpu().detach().numpy().copy()

        img_height = img_ref.shape[0]
        img_width = img_ref.shape[1]

        # perform skeletonization, need to extend the boundary of the image
        extend_dim = int(60)
        img_thresh_extend = np.zeros((img_height, img_width + extend_dim))
        img_thresh_extend[0:img_height, 0:img_width] = img_ref / 1.0

        left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, img_width - 1]))
        left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, img_width - 10]))

        extend_vec_pt1_center = np.array([img_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1]) / 2])
        extend_vec_pt2_center = np.array(
            [img_width - 5, (left_boundarylineB_id[0] + left_boundarylineB_id[-1]) / 2])
        exten_vec = extend_vec_pt2_center - extend_vec_pt1_center

        if exten_vec[1] == 0:
            exten_vec[1] += 0.00000001

        k_extend = exten_vec[0] / exten_vec[1]
        b_extend_up = img_width - k_extend * left_boundarylineA_id[0]
        b_extend_dw = img_width - k_extend * left_boundarylineA_id[-1]

        # then it could be able to get the intersection point with boundary
        extend_ROI = np.array([
            np.array([img_width, left_boundarylineA_id[0]]),
            np.array([img_width, left_boundarylineA_id[-1]]),
            np.array([img_width + extend_dim,
                      int(((img_width + extend_dim) - b_extend_dw) / k_extend)]),
            np.array([img_width + extend_dim,
                      int(((img_width + extend_dim) - b_extend_up) / k_extend)])
        ])

        img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

        skeleton = skimage_morphology.skeletonize(img_thresh_extend)

        img_raw_skeleton = np.argwhere(skeleton[:, 0:img_width] == 1)

        self.img_raw_skeleton = torch.as_tensor(img_raw_skeleton).float()


