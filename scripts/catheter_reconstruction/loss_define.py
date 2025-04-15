"""
File to define each loss function that we will be using. 
There are three loss functions: contour chamfer, tip distance, 
and loss function between two images of a catheter. 
"""
import sys
sys.path.append('..')
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import numpy as np
import torch
import torch.nn as nn
import cv2
import skimage.morphology as skimage_morphology

class GenerateRefData(): 
    '''
    Class used to generate reference data for the catheter reconstruction model.
    '''
    def __init__(self, img_ref):
        self.img_ref = img_ref
        
    def find_endpoints(self, skeleton, skeleton_coords):
        """
        Find endpoints of the skeleton and their indices
        """
        endpoints = []
        endpoint_indices = []
        for idx, (x, y) in enumerate(skeleton_coords):
            neighborhood = skeleton[x-1:x+2, y-1:y+2]
            if np.sum(neighborhood) == 2:  # endpoint will have only one neighbor in the skeleton
                endpoints.append((x, y))
                endpoint_indices.append(idx)

        if len(endpoints) != 2:
            raise ValueError("The skeleton does not have exactly two endpoints.")

        # Determine tip and base based on y-coordinate
        if endpoints[0][1] < endpoints[1][1]:
            tip, base = endpoints[0], endpoints[1]
            tip_idx, base_idx = endpoint_indices[0], endpoint_indices[1]
        else:
            tip, base = endpoints[1], endpoints[0]
            tip_idx, base_idx = endpoint_indices[1], endpoint_indices[0]

        # Swap tip with the first element and base with the last element
        skeleton_coords[0], skeleton_coords[tip_idx] = skeleton_coords[tip_idx], skeleton_coords[0]
        skeleton_coords[-1], skeleton_coords[base_idx] = skeleton_coords[base_idx], skeleton_coords[-1]

        return skeleton_coords

    def get_raw_centerline(self):
        '''
        Method to get the raw centerline of the catheter from the reference image.
        '''

        # convert to numpy array
        img_ref = self.img_ref.cpu().detach().numpy().copy()

        img_height = img_ref.shape[0]
        img_width = img_ref.shape[1]

        # perform skeletonization, need to extend the boundary of the image because of the way the skeletonization algorithm works (it looks at the 8 neighbors of each pixel)
        extend_dim = int(60)
        img_thresh_extend = np.zeros((img_height, img_width + extend_dim))
        img_thresh_extend[0:img_height, 0:img_width] = img_ref / 1.0

        # get the left boundary of the image
        left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, img_width - 1]))
        left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, img_width - 10]))

        # get the center of the left boundary
        extend_vec_pt1_center = np.array([img_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1]) / 2])
        extend_vec_pt2_center = np.array(
            [img_width - 5, (left_boundarylineB_id[0] + left_boundarylineB_id[-1]) / 2])
        exten_vec = extend_vec_pt2_center - extend_vec_pt1_center

        # avoid dividing by zero
        if exten_vec[1] == 0:
            exten_vec[1] += 0.00000001

        # get the slope and intercept of the line
        k_extend = exten_vec[0] / exten_vec[1]
        b_extend_up = img_width - k_extend * left_boundarylineA_id[0]
        b_extend_dw = img_width - k_extend * left_boundarylineA_id[-1]

        # extend the ROI to the right, so that the skeletonization algorithm could be able to get the centerline
        # then it could be able to get the intersection point with boundary
        extend_ROI = np.array([
            np.array([img_width, left_boundarylineA_id[0]]),
            np.array([img_width, left_boundarylineA_id[-1]]),
            np.array([img_width + extend_dim,
                      int(((img_width + extend_dim) - b_extend_dw) / k_extend)]),
            np.array([img_width + extend_dim,
                      int(((img_width + extend_dim) - b_extend_up) / k_extend)])
        ])

        # fill the extended ROI with 1
        img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

        # skeletonize the image
        skeleton = skimage_morphology.skeletonize(img_thresh_extend)
        skeleton[:, img_width:] = 0

        # get the centerline of the image
        img_raw_skeleton = np.argwhere(skeleton[:, 0:img_width] == 1)
        img_raw_skeleton = self.find_endpoints(skeleton, img_raw_skeleton)

        self.img_raw_skeleton = torch.as_tensor(img_raw_skeleton).float()

        return self.img_raw_skeleton
        
    def get_raw_contour(self): 
        '''
        Method to compute the raw contour of the catheter from the reference image.
        '''

        # Convert reference image to numpy array of type np.uint8 to be able to use OpenCV
        img_ref = self.img_ref.cpu().detach().numpy().copy().astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(img_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the largest contour (assuming it's the tube)
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract coordinates of contour pixels
        ref_catheter_contour_coordinates = largest_contour.squeeze()

        # Convert coordinates to PyTorch tensor
        self.ref_catheter_contour_point_cloud = torch.tensor(ref_catheter_contour_coordinates, dtype=torch.float)

        return self.ref_catheter_contour_point_cloud


class ContourChamferLoss(nn.Module):
    '''
    Class to define contour chamfer loss.
    '''

    def __init__(self, device):
        super(ContourChamferLoss, self).__init__()
        self.device = device

    def forward(self, img_render_points, ref_catheter_contour_point_cloud):
        """
        Calculate the Chamfer loss between projected points and reference image's catheter contour.
        
        Args:
            img_render_points (Tensor): Image of projected points. Must reshape to shape (N, 2).
            ref_catheter_contour_point_cloud (Tensor): Reference contour of catheter 
                                                       (coordinates of pixels on catheter border), 
                                                       Reshape to shape (# of pixels inside the contour , 2).      
        Returns:
            loss (Tensor): Contour Chamfer loss.
        """

        # reshape img_render_points to shape (img_render_points.shape[0] * img_render_points.shape[1], 2)
        self.img_render_point_cloud = img_render_points.reshape(img_render_points.shape[0] * img_render_points.shape[1], 2)
        # print("self.img_render_point_cloud shape: ", self.img_render_point_cloud.shape)
        # print("self.img_render_point_cloud: ", self.img_render_point_cloud)
        
        mask = (self.img_render_point_cloud[:, 0] >= 0) & (self.img_render_point_cloud[:, 0] <= 2000) & \
            (self.img_render_point_cloud[:, 1] >= 0) & (self.img_render_point_cloud[:, 1] <= 1000)
        self.img_render_point_cloud = self.img_render_point_cloud[mask]
        
        # downsample of projected point cloud
        num_points = self.img_render_point_cloud.shape[0]
        target_num_points = int(num_points/8)
        indices = torch.linspace(0, num_points - 1, target_num_points).long()
        self.img_render_point_cloud = self.img_render_point_cloud[indices]
        
        # downsample of reference point cloud
        num_points = ref_catheter_contour_point_cloud.shape[0]
        target_num_points = int(num_points/8)
        indices = torch.linspace(0, num_points - 1, target_num_points).long()
        ref_catheter_contour_point_cloud = ref_catheter_contour_point_cloud[indices]

        # Calculate pairwise Euclidean distances
        distances = torch.norm(self.img_render_point_cloud[:, None, :] - ref_catheter_contour_point_cloud[None, :, :], dim=2)

        # print("distances.shape: ", distances.shape)
        # print("distances: ", distances)
        
        # Find the minimum distance for each point in points1
        min_distances_1 = torch.min(distances, dim=1)[0]

        # print("min_distances_1.shape: ", min_distances_1.shape)
        # print("min_distances_1: ", min_distances_1)
        
        # Find the minimum distance for each point in points2
        min_distances_2 = torch.min(distances, dim=0)[0]

        # print("min_distances_2.shape: ", min_distances_2.shape)
        # print("min_distances_2: ", min_distances_2)
        
        # Calculate Chamfer loss
        # chamfer_loss = torch.sum(min_distances_1) + torch.sum(min_distances_2)
        chamfer_loss = (torch.mean(min_distances_1) + torch.mean(min_distances_2)) / 2
        # print("chamfer_loss: ", chamfer_loss)

        return chamfer_loss

class TipDistanceLoss(nn.Module): 
    '''
    Class to define the tip distance loss function.
    '''

    def __init__(self, device): 
        super(TipDistanceLoss, self).__init__()
        self.device = device

    def forward(self, img_render_centerline_points, ref_catheter_skeleton): 
        '''
        Calculate the Squared Distance loss between projected tip point (projected last centerline point) 
            and reference image's catheter tip.
        
        Args:
            img_render_centerline_points (Tensor): tensor of projected centerline points, Shape: (N, 2). 
                                                   Only use last projected 'point' as tip. 
            ref_catheter_skeleton (Tensor): Reference morphological skeleton of catheter. 
                              Must get coordinates of pixels at the tip, 
                              and reshape to shape (# of pixels inside the contour , 2).      
        Returns:
            loss (Tensor): Tip Squared Distance loss.

        '''

        # Flip the x and y coordinates in self.img_raw_skeleton: i.e., [[69, 43]] -> [[43, 69]]
        self.img_raw_skeleton = ref_catheter_skeleton.flip(1)
        ref_skeleton_tip_point = self.img_raw_skeleton[0, :]
        
        proj_tip_point = img_render_centerline_points[-1, :]
        # print("proj_tip_point: ", proj_tip_point)

        # Compute squared distance loss
        tip_distance_loss = torch.mean((proj_tip_point - ref_skeleton_tip_point) ** 2)

        # Compute euclidean distance loss (not used in loss computation -- only for figure generation)
        tip_euclidean_distance_loss = torch.norm(proj_tip_point - ref_skeleton_tip_point, p=2).detach().cpu().numpy().copy()


        return tip_distance_loss, tip_euclidean_distance_loss

class ImageContourChamferLoss(nn.Module): 
    '''
    Class to define the contour chamfer loss function between two images.
    '''
    def __init__(self, device): 
        super(ImageContourChamferLoss, self).__init__()
        self.device = device

    def forward(self): 
        '''
        Calculate the Chamfer loss between reconstructed image's catheter contour 
        and reference image's catheter contour. 
        
        Args:
        
        Returns:
            loss (Tensor): Image Contour Chamfer Loss.

        '''

        loss = 1


        return loss
    

class CenterlineLoss(nn.Module): 
    '''
    Class to define the contour chamfer loss function between two images.
    '''
    def __init__(self, device): 
        super(CenterlineLoss, self).__init__()
        self.device = device

    def forward(self, bezier_proj_centerline_img, ref_catheter_centerline): 
        '''

        '''
        ref_catheter_centerline = ref_catheter_centerline.flip(1)
        bezier_proj_centerline_img = bezier_proj_centerline_img.flip(0)
        # print(bezier_proj_centerline_img)
        
        mask_proj = (bezier_proj_centerline_img[:, 0] >= -2000) & (bezier_proj_centerline_img[:, 0] <= 2000) & \
            (bezier_proj_centerline_img[:, 1] >= -2000) & (bezier_proj_centerline_img[:, 1] <= 2000)
        bezier_proj_centerline_img = bezier_proj_centerline_img[mask_proj]
        
        # # downsample
        # target_num_points = bezier_proj_centerline_img.shape[0]
        
        # indices_ref = torch.linspace(0, ref_catheter_centerline.shape[0] - 1, target_num_points).long()
        # ref_catheter_centerline = ref_catheter_centerline[indices_ref]
        
        # distances = torch.sqrt(torch.sum((bezier_proj_centerline_img - ref_catheter_centerline) ** 2, dim=1))
        # average_distance = torch.mean(distances)
        
        # Calculate pairwise Euclidean distances
        distances = torch.norm(bezier_proj_centerline_img[:, None, :] - ref_catheter_centerline[None, :, :], dim=2)        
        # Find the minimum distance for each point in points1
        min_distances_1 = torch.min(distances, dim=1)[0]        
        # Find the minimum distance for each point in points2
        min_distances_2 = torch.min(distances, dim=0)[0]        
        # Calculate Chamfer loss
        centerline_loss = (torch.mean(min_distances_1) + torch.mean(min_distances_2)) / 2

        return centerline_loss

