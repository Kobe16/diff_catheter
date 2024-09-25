'''
File used to define custom loss functions for catheter reconstruction.
'''

from matplotlib import pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

import skimage.morphology as skimage_morphology
import cv2

from test_reconst_v2 import ConstructionBezier


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
        

class ChamferLossWholeImage(nn.Module):
    '''
    Class to define chamfer loss on image grid (Used to ensure projected 
    points fall within the image).
    '''

    def __init__(self, device):
        super(ChamferLossWholeImage, self).__init__()
        self.device = device

    def forward(self, img_render_points, img_ref):
        """
        Calculate the Chamfer loss between rendered points and grid of points
            created by reference image.
        
        Args:
            img_render_points (Tensor): Image of projected points. Must reshape to shape (N, 2).
            img_ref (Tensor): Reference image of catheter. 
                              Must create coordinate grid and reshape to shape (width * height, 2).      
        Returns:
            loss (Tensor): Chamfer loss.
        """
        self.prepare_data(img_render_points, img_ref)

        # Calculate pairwise Euclidean distances
        distances = torch.norm(self.img_render_point_cloud[:, None, :] - self.coordinates_point_cloud[None, :, :], dim=2)

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
        chamfer_loss = torch.sum(min_distances_1) + torch.sum(min_distances_2)
        # print("chamfer_loss: ", chamfer_loss)

        
        return chamfer_loss

    def prepare_data(self, img_render_points, img_ref): 
        '''
        Prepare data for Chamfer loss calculation.
        '''

        # Height = 480, Width = 640
        self.height = img_ref.shape[0]
        self.width = img_ref.shape[1]

        # reshape img_render_points to shape (img_render_points.shape[0] * img_render_points.shape[1], 2)
        self.img_render_point_cloud = img_render_points.reshape(img_render_points.shape[0] * img_render_points.shape[1], 2)
        # print("self.img_render_point_cloud shape: ", self.img_render_point_cloud.shape)
        # print("self.img_render_point_cloud: ", self.img_render_point_cloud)


        # Create point cloud for reference image dimensions
        # Create grid of coordinates using torch.meshgrid()
        x = torch.arange(0, self.width, 10)
        y = torch.arange(0, self.height, 10)
        xx, yy = torch.meshgrid(x, y)

        # Reshape and stack the coordinates
        self.coordinates_point_cloud = torch.stack((yy, xx), dim=-1).reshape(-1, 2)


# class ContourChamferLoss(nn.Module):
#     '''
#     Class to define contour chamfer loss.
#     '''

#     def __init__(self, device):
#         super(ContourChamferLoss, self).__init__()
#         self.device = device

#     def forward(self, img_render_points, ref_catheter_contour_point_cloud):
#         """
#         Calculate the Chamfer loss between projected points and reference image's catheter contour.
        
#         Args:
#             img_render_points (Tensor): Image of projected points. Must reshape to shape (N, 2).
#             ref_catheter_contour_point_cloud (Tensor): Reference contour of catheter 
#                                                        (coordinates of pixels on catheter border), 
#                                                        Reshape to shape (# of pixels inside the contour , 2).      
#         Returns:
#             loss (Tensor): Contour Chamfer loss.
#         """

#         # reshape img_render_points to shape (img_render_points.shape[0] * img_render_points.shape[1], 2)
#         self.img_render_point_cloud = img_render_points.reshape(img_render_points.shape[0] * img_render_points.shape[1], 2)
#         # print("self.img_render_point_cloud shape: ", self.img_render_point_cloud.shape)
#         # print("self.img_render_point_cloud: ", self.img_render_point_cloud)

#         # Calculate pairwise Euclidean distances
#         distances = torch.norm(self.img_render_point_cloud[:, None, :] - ref_catheter_contour_point_cloud[None, :, :], dim=2)

#         # print("distances.shape: ", distances.shape)
#         # print("distances: ", distances)
        
#         # Find the minimum distance for each point in points1
#         min_distances_1 = torch.min(distances, dim=1)[0]

#         # print("min_distances_1.shape: ", min_distances_1.shape)
#         # print("min_distances_1: ", min_distances_1)
        
#         # Find the minimum distance for each point in points2
#         min_distances_2 = torch.min(distances, dim=0)[0]

#         # print("min_distances_2.shape: ", min_distances_2.shape)
#         # print("min_distances_2: ", min_distances_2)
        
#         # Calculate Chamfer loss
#         chamfer_loss = torch.sum(min_distances_1) + torch.sum(min_distances_2)
#         # print("chamfer_loss: ", chamfer_loss)

#         return chamfer_loss

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
        # self.img_render_point_cloud = img_render_points
        # print("self.img_render_point_cloud shape: ", self.img_render_point_cloud.shape)
        # print("self.img_render_point_cloud: ", self.img_render_point_cloud)
        
        mask = (self.img_render_point_cloud[:, 0] >= 0) & (self.img_render_point_cloud[:, 0] <= 640) & \
            (self.img_render_point_cloud[:, 1] >= 0) & (self.img_render_point_cloud[:, 1] <= 480)
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


class TipChamferLoss(nn.Module):
    '''
    Class to define tip chamfer loss.
    '''
    def __init__(self, device):
        super(TipChamferLoss, self).__init__()
        self.device = device

    def forward(self, img_render_points, ref_catheter_skeleton):
        """
        Calculate the Chamfer loss between projected points LAST circle and reference image's catheter tip.
        
        Args:
            img_render_points (Tensor): Image of projected points. Must reshape to shape (N, 2). 
                                        Only use last projected 'circle'. 
            ref_catheter_skeleton (Tensor): Reference morphological skeleton of catheter. 
                              Must get coordinates of pixels at the tip, 
                              and reshape to shape (# of pixels inside the contour , 2).      
        Returns:
            loss (Tensor): Tip Chamfer loss.
        """
        self.prepare_data(img_render_points, ref_catheter_skeleton)

        # Calculate pairwise Euclidean distances
        distances = torch.norm(self.img_render_points_last_circle_point_cloud[:, None, :] - self.ref_catheter_tip_point_cloud[None, :, :], dim=2)

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
        chamfer_loss = torch.sum(min_distances_1) + torch.sum(min_distances_2)
        # print("chamfer_loss: ", chamfer_loss)

        
        return chamfer_loss
    
    def prepare_data(self, img_render_points, ref_catheter_skeleton):
        '''
        Method to prepare data for chamfer loss calculation.
        '''

        # Extract last circle of projected points
        img_render_points_last_circle = img_render_points[-1]

        # reshape img_render_points_last_circle to shape (# of points in 1 circle, 2)
        self.img_render_points_last_circle_point_cloud = img_render_points_last_circle.reshape(img_render_points_last_circle.shape[0], 2)
        # print("self.img_render_points_last_circle_point_cloud shape: ", self.img_render_points_last_circle_point_cloud.shape)
        # print("self.img_render_points_last_circle_point_cloud: ", self.img_render_points_last_circle_point_cloud)


        # Extract tip point on catheter skeleton. Then add a dimension to make it
        # technically count as a point cloud (shape (1, 2))
        # print("ref_catheter_skeleton.shape: ", ref_catheter_skeleton.shape)
        # print("ref_catheter_skeleton: ", ref_catheter_skeleton)
        self.ref_catheter_tip_point_cloud = ref_catheter_skeleton[0]
        self.ref_catheter_tip_point_cloud = self.ref_catheter_tip_point_cloud.unsqueeze(0)

        # Flip the x and y coordinates in self.ref_catheter_tip_point_cloud: i.e., [[69, 43]] -> [[43, 69]]
        self.ref_catheter_tip_point_cloud = self.ref_catheter_tip_point_cloud.flip(1)

        # print("self.ref_catheter_tip_point_cloud.shape: ", self.ref_catheter_tip_point_cloud.shape)
        # print("self.ref_catheter_tip_point_cloud: ", self.ref_catheter_tip_point_cloud)


class BoundaryPointChamferLoss(nn.Module):
    '''
    Class to define boundary point chamfer loss. 
    '''

    def __init__(self, device):
        super(BoundaryPointChamferLoss, self).__init__()
        self.device = device

    def forward(self, img_render_points, ref_catheter_skeleton):
        """
        Calculate the Chamfer loss between projected points First circle and reference image's boundary point.
        
        Args:
            img_render_points (Tensor): Image of projected points. Must reshape to shape (N, 2). 
                                        Only use first projected 'circle'. 
            ref_catheter_skeleton (Tensor): Reference morphological skeleton of catheter. 
                              Must get coordinates of pixels at the boundary, 
                              and reshape to shape (# of pixels inside the contour , 2).      
        Returns:
            loss (Tensor): Tip Chamfer loss.
        """
        self.prepare_data(img_render_points, ref_catheter_skeleton)

        # Calculate pairwise Euclidean distances
        distances = torch.norm(self.img_render_points_first_circle_point_cloud[:, None, :] - self.ref_catheter_boundary_point_cloud[None, :, :], dim=2)

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
        chamfer_loss = torch.sum(min_distances_1) + torch.sum(min_distances_2)
        # print("chamfer_loss: ", chamfer_loss)

        
        return chamfer_loss
    
    def prepare_data(self, img_render_points, ref_catheter_skeleton):
        '''
        Method to prepare data for chamfer loss calculation.
        '''

        # Extract last circle of projected points
        img_render_points_first_circle = img_render_points[0]

        # reshape img_render_points_last_circle to shape (# of points in 1 circle, 2)
        self.img_render_points_first_circle_point_cloud = img_render_points_first_circle.reshape(img_render_points_first_circle.shape[0], 2)
        # print("self.img_render_points_last_circle_point_cloud shape: ", self.img_render_points_last_circle_point_cloud.shape)
        # print("self.img_render_points_last_circle_point_cloud: ", self.img_render_points_last_circle_point_cloud)


        # Extract tip point on catheter skeleton. Then add a dimension to make it
        # technically count as a point cloud (shape (1, 2))
        # print("self.img_raw_skeleton.shape: ", self.img_raw_skeleton.shape)
        # print("self.img_raw_skeleton: ", self.img_raw_skeleton)
        self.ref_catheter_boundary_point_cloud = ref_catheter_skeleton[-1]
        self.ref_catheter_boundary_point_cloud = self.ref_catheter_boundary_point_cloud.unsqueeze(0)

        # Flip the x and y coordinates in self.ref_catheter_tip_point_cloud: i.e., [[69, 43]] -> [[43, 69]]
        self.ref_catheter_boundary_point_cloud = self.ref_catheter_boundary_point_cloud.flip(1)

        # print("self.ref_catheter_tip_point_cloud.shape: ", self.ref_catheter_tip_point_cloud.shape)
        # print("self.ref_catheter_tip_point_cloud: ", self.ref_catheter_tip_point_cloud)


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


class BoundaryPointDistanceLoss(nn.Module): 
    '''
    Class to define the boundary point distance loss.
    '''

    def __init__(self, device): 
        super(BoundaryPointDistanceLoss, self).__init__()
        self.device = device

    def forward(self, img_render_centerline_points, ref_catheter_skeleton): 
        '''
        Calculate the Squared Distance loss between projected base point (projected first centerline point) 
            and reference image's catheter boundary point.
        
        Args:
            img_render_centerline_points (Tensor): tensor of projected centerline points, Shape: (N, 2). 
                                                   Only use first projected 'point' as base point. 
            ref_catheter_skeleton (Tensor): Reference morphological skeleton of catheter. 
                              Must get coordinates of pixels at the boundary, 
                              and reshape to shape (# of pixels inside the contour , 2).      
        Returns:
            loss (Tensor): Boundary Point Squared Distance loss.

        '''
        
        # Flip the x and y coordinates in self.img_raw_skeleton: i.e., [[69, 43]] -> [[43, 69]]
        self.img_raw_skeleton = ref_catheter_skeleton.flip(1)

        ref_boundary_point = self.img_raw_skeleton[-1, :]
        # print("ref_tip_point: ", ref_boundary_point)

        proj_boundary_point = img_render_centerline_points[0, :]
        # print("proj_tip_point: ", proj_boundary_point)

        tip_distance_loss = torch.mean((proj_boundary_point - ref_boundary_point) ** 2)

        return tip_distance_loss
    
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
        
        mask_proj = (bezier_proj_centerline_img[:, 0] >= 0) & (bezier_proj_centerline_img[:, 0] <= 640) & \
            (bezier_proj_centerline_img[:, 1] >= 0) & (bezier_proj_centerline_img[:, 1] <= 480)
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
  

if __name__ == '__main__': 
    '''
    Main function used to test out loss functions. Doesn't do anything in grand scheme of things, 
    just used for testing to get stuff right. 
    '''

    ###========================================================
    ### 1) SET TO GPU OR CPU COMPUTING
    ###========================================================
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device("cuda:0")
        torch.cuda.set_device(gpu_or_cpu)
    else:
        gpu_or_cpu = torch.device("cpu")

    ###========================================================
    ### 2) VARIABLES FOR BEZIER CURVE CONSTRUCTION
    ###========================================================
    para_init = torch.tensor([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866], dtype=torch.float, requires_grad=False)
    p_start = torch.tensor([0.02, 0.002, 0.0])

    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/diff_render_1'
    img_save_path = case_naming + '.png'

    '''
    Create binary mask of catheter: 
        1) Grayscale the ref img, 
        2) threshold the grayscaled img, 
        3) Creates a binary image by replacing all 
           pixel values equal to 255 with 1 (leaves
           other pixel values unchanged)
    '''
    img_ref_rgb = cv2.imread(img_save_path)
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)
    (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)


    ###========================================================
    ### 3) SETTING UP BEZIER CURVE CONSTRUCTION
    ###========================================================
    build_bezier = ConstructionBezier()
    build_bezier.loadRawImage(img_save_path)


    ###========================================================
    ### 4) RUNNING BEZIER CURVE CONSTRUCTION
    ###========================================================
    # Generate the Bezier curve cylinder mesh points
    build_bezier.getBezierCurveCylinder(p_start, para_init)

    # Plot 3D Bezier Cylinder mesh points
    build_bezier.plot3dBezierCylinder()

    # Plot 2D projected Bezier Cylinder mesh points
    build_bezier.getCylinderMeshProjImg()
    build_bezier.draw2DCylinderImage()


    ###========================================================
    ### 4) GET BINARIZED IMAGE OF PROJECTED POINTS
    ###========================================================

    img_proj_pts_bin = build_bezier.get2DCylinderImage()

    # print("img_proj_pts_bin: " + str(img_proj_pts_bin))

    ###========================================================
    ### 4) TEST OUT ANY LOSS FUNCTION 
    ###========================================================

