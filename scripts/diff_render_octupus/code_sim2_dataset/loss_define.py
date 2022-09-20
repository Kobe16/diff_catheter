import torch

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap
import matplotlib.pyplot as plt

import skimage.morphology as skimage_morphology

from sklearn.neighbors import NearestNeighbors
import networkx as nx

import skfmm

import cv2

import pdb

import numpy as np


class ContourLoss(nn.Module):

    def __init__(self, device):
        super(ContourLoss, self).__init__()
        self.device = device

    def forward(self, img_render_original, img_ref, img_ref_dist_map):

        # Binarize img_render [0.0, 1.0] -> {0., 1.}
        # img_render = (img_render >= 0.1).float()   # Thresholding is NOT differentiable\

        thresholding = 0.1
        img_render_diffable = 1 / (1 + torch.exp(-100 * (img_render_original - thresholding)))  # Differentiable binarization (approximation)

        mask = (img_render_diffable < thresholding)
        img_render_mask = img_render_diffable * mask  # Zero out values above the threshold 0.1

        # pdb.set_trace()

        # Convert from (B x H x W) to (B x C x H x W)
        img_render_diffable = torch.unsqueeze(img_render_diffable, 1)

        # Apply Laplacian operator to grayscale images to find contours
        kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]).to(self.device)

        contours = F.conv2d(img_render_diffable, kernel, padding=1)
        contours = torch.clamp(contours, min=0, max=255)

        # Convert from (B x C x H x W) back to (B x H x W)
        contours = torch.squeeze(contours, 1)

        # Compute the Chamfer distance between two images
        # Selecting indices is NOT differentiable -> use tanh(x) or 2 / (1 + e^(-100(x))) - 1 for differentiable thresholding
        # -> apply element-wise product between contours and distance maps
        img_render_contour = torch.tanh(contours)

        diff_dist = img_render_contour * img_ref_dist_map  # element-wise product
        # diff_dist = img_render_diffable * img_ref_dist_map  # element-wise product

        # dist = diff_dist.sum() / (img_render_contour.shape[1] + img_render_contour.shape[2])
        dist = diff_dist.sum() / 1.0
        assert (dist >= 0)

        # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        # ax = axes.ravel()
        # ax[0].imshow(img_ref_dist_map.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        # ax[0].set_title('img ref binarized')
        # ax[1].imshow(img_render_diffable.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        # ax[1].set_title('render thresholding')
        # ax[2].imshow(img_render_contour.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        # ax[2].set_title('render thresholding')
        # ax[3].imshow(img_render_original.squeeze().cpu().detach().numpy(), cmap=colormap.gray)
        # ax[3].set_title('difference')
        # plt.show()

        # pdb.set_trace()

        return dist, img_render_contour.squeeze(), img_render_diffable.squeeze()


class MaskLoss(nn.Module):

    def __init__(self, device):
        super(MaskLoss, self).__init__()
        self.device = device

    def forward(self, img_render, img_ref):
        # Binarize img_render [0.0, 1.0] -> {0., 1.}
        # img_render = (img_render >= 0.1).float()   # Thresholding is NOT differentiable

        # img_render = 1 / (1 + torch.exp(-100 * (img_render - 0.1)))  # Differentiable binarization (approximation)
        # mask = (img_render > 0.1)
        # img_render = img_render * mask  # Zero out values above the threshold 0.5

        # img_render_binary = img_render.squeeze()

        # img_diff = torch.abs(img_render - img_ref)**2

        dist = torch.sum((img_render - img_ref)**2)

        # pdb.set_trace()

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

        return dist


class CenterlineLoss(nn.Module):

    def __init__(self, device):
        super(CenterlineLoss, self).__init__()
        self.device = device

        self.img_raw_skeleton = None

    def forward(self, bezier_proj_img, img_ref, selected_frame_id):

        opt_skeleton_ordered_all = self.get_raw_centerline(img_ref)

        ## 0-93  / 1-94  / 2-95  / 3-97  / 4-98/
        ## 5-104 / 6-105 / 7-108 / 8-109 / 9-110/
        ## 10-111 / 11-112 / 12-113 / 13-114 / 14-115
        ## 15-116 / 16-117 / 17-118 / 18-119 / 19-120
        ## 20-121 / 21-122 / 22-123 / 23-124 / 24-125
        ## 25-135 / 26-136 / 27-137 / 28-138 / 29-139
        ## 30-140 / 31-147
        crop_selected_skeleton = ( (10, -35), (0, -32),  (12, -35), (10, -35), (10, -62), (0, 60),   (0, 60),   (15, -45), (15, -35),  \
                                 (15, -50), (15, -34), (15, -26), (15, -26), (15, -26), (15, -26), (15, -20), (15, -20), (15, -15), \
                                 (15, -12), (15, -1),  (15, -1),  (15, -1),  (15, -1),  (15, -1),  (15, -1),  (15, -1),  (15, -1),    \
                                 (15, -1),  (15, -1),  (15, -1),  (15, -1),  (15, -1))

        # (15,-45)

        # id_start = 15
        # id_end = -50
        # selected_frame_id = 0

        id_start = crop_selected_skeleton[selected_frame_id][0]
        id_end = crop_selected_skeleton[selected_frame_id][1]

        opt_skeleton_ordered = opt_skeleton_ordered_all[id_start:id_end, :]

        # pdb.set_trace()

        # # ---------------
        # # # plot with
        # # # ---------------
        # fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        # ax = axes.ravel()
        # ax[0].imshow(cv2.cvtColor(img_ref.cpu().detach().numpy(), cv2.COLOR_BGR2RGB))
        # ax[0].plot(opt_skeleton_ordered[:, 0], opt_skeleton_ordered[:, 1], marker='o', color='#E94560', markersize=1, linestyle='-', linewidth=0.1)
        # ax[0].set_title('skeleton')
        # plt.tight_layout()
        # plt.show()
        # pdb.set_trace()

        self.img_raw_skeleton_ordered = torch.as_tensor(opt_skeleton_ordered).float().to(self.device)

        diff_skeleton = torch.diff(self.img_raw_skeleton_ordered, axis=0)
        dis_diff_skeleton = torch.linalg.norm(diff_skeleton, ord=None, axis=1)
        dis_sum_skeleton = torch.cumsum(dis_diff_skeleton, dim=0)
        dis_sum_skeleton = torch.cat((torch.tensor([0]).to(self.device), dis_sum_skeleton), dim=0)
        dis_sum_skeleton = dis_sum_skeleton / dis_sum_skeleton[-1]

        diff_bezier = torch.diff(bezier_proj_img, axis=0)
        dis_diff_bezier = torch.linalg.norm(diff_bezier, ord=None, axis=1)
        dis_sum_bezier = torch.cumsum(dis_diff_bezier, dim=0)
        dis_sum_bezier = torch.cat((torch.tensor([0]).to(self.device), dis_sum_bezier), dim=0)
        dis_sum_bezier = dis_sum_bezier / dis_sum_bezier[-1]

        num_bzr = dis_sum_bezier.shape[0] - 1
        centerline_selected_id_list = (int(np.floor(num_bzr / 4)), int(np.floor(num_bzr / 4) * 2), int(np.floor(num_bzr / 4) * 3), num_bzr)

        skeleton_by_dis = []
        ref_skeleton_selected_id_list = []
        for i in range(len(centerline_selected_id_list)):
            err = torch.abs(dis_sum_bezier[centerline_selected_id_list[i]] - dis_sum_skeleton)
            index = torch.argmin(err)
            temp = self.img_raw_skeleton_ordered[index, :]
            skeleton_by_dis.append(temp)
            ref_skeleton_selected_id_list.append(index.cpu().detach().numpy())
        skeleton_by_dis = torch.stack(skeleton_by_dis)

        err_centerline = torch.linalg.norm(skeleton_by_dis - bezier_proj_img[centerline_selected_id_list[:], :], ord=None, axis=1)
        loss_centerline = torch.sum(err_centerline)

        # pdb.set_trace()
        return loss_centerline, opt_skeleton_ordered, centerline_selected_id_list, ref_skeleton_selected_id_list

    def get_raw_centerline(self, img_ref):
        ## https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line

        img_ref = img_ref.cpu().detach().numpy().copy()

        # img_height = img_ref.shape[0]
        # img_width = img_ref.shape[1]

        skeleton = skimage_morphology.skeletonize(img_ref)
        img_raw_skeleton = np.argwhere(skeleton == 1)

        # creating a nearest neighbour graph to connect each of the nodes to its 2 nearest neighbors
        # neigh = NearestNeighbors(n_neighbors=2, radius=0.4)

        clf = NearestNeighbors(n_neighbors=2).fit(img_raw_skeleton)
        G = clf.kneighbors_graph()

        # then use networkx to construct a graph from this sparse matrix
        T = nx.from_scipy_sparse_matrix(G)

        # find shortest path from source
        # minimizes the distances between the connections (optimization problem):
        min_dist = np.inf
        min_idx = 0
        opt_skeleton_ordered = None

        for i in range(img_raw_skeleton.shape[0]):
            path = list(nx.dfs_preorder_nodes(T, i))
            ordered = img_raw_skeleton[path]  # ordered nodes

            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
            if cost < min_dist:
                min_dist = cost
                min_idx = i
                opt_skeleton_ordered = ordered

        ### this can gurantee the starting point of the skeleton is always from top-->bottom
        if opt_skeleton_ordered[0, 0] > 50:
            opt_skeleton_ordered = np.flip(opt_skeleton_ordered, 0)

        ### this will flip x/y coordinates in order to fit bezier_proj_img
        opt_skeleton_ordered = np.stack((opt_skeleton_ordered[:, 1], opt_skeleton_ordered[:, 0]), axis=1)

        return opt_skeleton_ordered

    # def get_raw_centerline_endo_view(self, img_ref):

    #     img_ref = img_ref.cpu().detach().numpy().copy()

    #     img_height = img_ref.shape[0]
    #     img_width = img_ref.shape[1]

    #     # perform skeletonization, need to extend the boundary of the image
    #     extend_dim = int(60)
    #     img_thresh_extend = np.zeros((img_height, img_width + extend_dim))
    #     img_thresh_extend[0:img_height, 0:img_width] = img_ref / 1.0

    #     left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, img_width - 1]))
    #     left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, img_width - 10]))

    #     extend_vec_pt1_center = np.array([img_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1]) / 2])
    #     extend_vec_pt2_center = np.array([img_width - 5, (left_boundarylineB_id[0] + left_boundarylineB_id[-1]) / 2])
    #     exten_vec = extend_vec_pt2_center - extend_vec_pt1_center

    #     if exten_vec[1] == 0:
    #         exten_vec[1] += 0.00000001

    #     k_extend = exten_vec[0] / exten_vec[1]
    #     b_extend_up = img_width - k_extend * left_boundarylineA_id[0]
    #     b_extend_dw = img_width - k_extend * left_boundarylineA_id[-1]

    #     # then it could be able to get the intersection point with boundary
    #     extend_ROI = np.array([
    #         np.array([img_width, left_boundarylineA_id[0]]),
    #         np.array([img_width, left_boundarylineA_id[-1]]),
    #         np.array([img_width + extend_dim,
    #                   int(((img_width + extend_dim) - b_extend_dw) / k_extend)]),
    #         np.array([img_width + extend_dim,
    #                   int(((img_width + extend_dim) - b_extend_up) / k_extend)])
    #     ])

    #     img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

    #     skeleton = skimage_morphology.skeletonize(img_thresh_extend)

    #     img_raw_skeleton = np.argwhere(skeleton[:, 0:img_width] == 1)

    #     self.img_raw_skeleton = torch.as_tensor(img_raw_skeleton).float()


class KeypointsIn3DLoss(nn.Module):

    def __init__(self, device):
        super(KeypointsIn3DLoss, self).__init__()
        self.device = device

    def forward(self, bezier_pos_3d, gt_skeleton_3d):

        # self.gt_skeleton_3d = torch.as_tensor(gt_skeleton_3d).float().to(self.device)

        # loss_startpoint = torch.linalg.norm(bezier_pos_3d[0, :] - self.gt_skeleton_3d[0, :], ord=None, axis=0)
        # loss_endpoint = torch.linalg.norm(bezier_pos_3d[-1, :] - self.gt_skeleton_3d[-1, :], ord=None, axis=0)

        # loss_endpoints_3d = loss_startpoint + loss_endpoint

        loss_endpoints_3d = 0

        return loss_endpoints_3d


class KeypointsInImageLoss(nn.Module):

    def __init__(self, device):
        super(KeypointsInImageLoss, self).__init__()
        self.device = device

    def forward(self, bezier_proj_img, bezier_der_proj_img, img_gt_skeleton):

        pt_intesection_endpoints_ref = self.get_intesection_endpoints_ref(img_gt_skeleton)
        pt_intesection_endpoints_render = self.get_intesection_endpoints_render(bezier_proj_img, bezier_der_proj_img)

        self.img_gt_skeleton = img_gt_skeleton

        # loss_startpoint = torch.linalg.norm(bezier_proj_img[0, :] - self.img_gt_skeleton[0, :], ord=None, axis=0)
        loss_endpoint = torch.linalg.norm(bezier_proj_img[-1, :] - self.img_gt_skeleton[-1, :], ord=None, axis=0)
        loss_intesection = torch.linalg.norm(pt_intesection_endpoints_render - pt_intesection_endpoints_ref, ord=None, axis=0)

        # loss_keypoints_img = loss_startpoint + loss_endpoint * 10 + loss_intesection * 10
        # loss_keypoints_img = loss_endpoint * 1 + loss_intesection * 1
        loss_keypoints_img = loss_endpoint

        # pdb.set_trace()

        return loss_keypoints_img, pt_intesection_endpoints_ref, pt_intesection_endpoints_render

    def get_intesection_endpoints_ref(self, img_gt_skeleton):
        ## https://math.stackexchange.com/questions/406864/intersection-of-two-lines-in-vector-form

        pt_start = img_gt_skeleton[0, :]
        pt_start_tang = img_gt_skeleton[1, :]
        vec_start = pt_start_tang - pt_start
        vec_start_norm = vec_start / torch.linalg.norm(vec_start, ord=None, axis=0)

        pt_end = img_gt_skeleton[-1, :]
        pt_end_tang = img_gt_skeleton[-2, :]
        vec_end = pt_end_tang - pt_end
        vec_end_norm = vec_end / torch.linalg.norm(vec_end, ord=None, axis=0)

        # pdb.set_trace()

        ## Ax=b
        mat_A = torch.cat((vec_start_norm.unsqueeze(1), -vec_end_norm.unsqueeze(1)), dim=1)
        mat_b = pt_end - pt_start

        mat_A_start = torch.clone(mat_A)
        mat_A_start[:, 0] = mat_b

        det_mat_A = torch.linalg.det(mat_A)
        det_mat_A_start = torch.linalg.det(mat_A_start)

        if det_mat_A != 0:
            scale_start = det_mat_A_start / det_mat_A
        else:
            scale_start = torch.linalg.norm(pt_end - pt_start, ord=None, axis=0) / 2
            print("=============================================================")
            print("This is a special intesection_endpoints_tangent case !!!")
            print("=============================================================")

        pt_intesection_endpoints_ref = pt_start + scale_start * vec_start_norm

        return pt_intesection_endpoints_ref

    def get_intesection_endpoints_render(self, bezier_proj_img, bezier_der_proj_img):
        ## https://math.stackexchange.com/questions/406864/intersection-of-two-lines-in-vector-form

        pt_start = bezier_proj_img[0, :]
        # pt_start_tang = bezier_proj_img[1, :]
        vec_start = bezier_der_proj_img[0, :]
        vec_start_norm = vec_start / torch.linalg.norm(vec_start, ord=None, axis=0)

        pt_end = bezier_proj_img[-1, :]
        # pt_end_tang = bezier_proj_img[-2, :]
        vec_end = bezier_der_proj_img[-1, :]
        vec_end_norm = vec_end / torch.linalg.norm(vec_end, ord=None, axis=0)

        ## Ax=b
        mat_A = torch.cat((vec_start_norm.unsqueeze(1), -vec_end_norm.unsqueeze(1)), dim=1)
        mat_b = pt_end - pt_start

        mat_A_start = torch.clone(mat_A)
        mat_A_start[:, 0] = mat_b

        det_mat_A = torch.linalg.det(mat_A)
        det_mat_A_start = torch.linalg.det(mat_A_start)

        if det_mat_A != 0:
            scale_start = det_mat_A_start / det_mat_A
        else:
            scale_start = torch.linalg.norm(pt_end - pt_start, ord=None, axis=0) / 2
            print("=============================================================")
            print("This is a special intesection_endpoints_tangent case !!!")
            print("=============================================================")

        pt_intesection_endpoints_render = pt_start + scale_start * vec_start_norm

        return pt_intesection_endpoints_render
