import torch

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap
import matplotlib.pyplot as plt

import pdb


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

    def forward(self, img_render, img_ref):
        # Binarize img_render [0.0, 1.0] -> {0., 1.}
        # img_render = (img_render >= 0.1).float()   # Thresholding is NOT differentiable
        img_render = 1 / (1 + torch.exp(-100 * (img_render - 0.1)))  # Differentiable binarization (approximation)
        mask = (img_render > 0.1)
        img_render = img_render * mask  # Zero out values above the threshold 0.5

        img_render_binary = img_render.squeeze()

        img_diff = torch.abs(img_render - img_ref)

        dist = img_diff.sum()
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
