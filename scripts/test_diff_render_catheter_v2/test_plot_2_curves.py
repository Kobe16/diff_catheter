import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = [1280, 800]
# mpl.rcParams['figure.dpi'] = 300

sys.path.insert(1, '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts')
import camera_settings

import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
# from skimage.morphology import skeletonize
import skimage.morphology as skimage_morphology


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from random import random, randrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
import pdb
import argparse

class ConstructionBezier(nn.Module):
    '''
    def __init__(self, img_path, curve_length_gt, P0_gt, para_gt, para_init, loss_weight, total_itr, verbose=0):

        # self.img_id = 1
        # self.save_dir = './steps_imgs_left_1_STCF'
        # if os.path.isdir(self.save_dir):
        #     shutil.rmtree(self.save_dir)
        # os.mkdir(self.save_dir)
        # os.mkdir(self.save_dir + '/centerline')
        # os.mkdir(self.save_dir + '/contours')

        self.curve_length_gt = curve_length_gt
        self.P0_gt = P0_gt
        self.para_gt = para_gt
        self.para = para_init
        self.loss_weight = loss_weight
        self.total_itr = total_itr
        self.verbose = verbose

        self.OFF_SET = torch.tensor([0.00, 0.00, 0.00])

        # self.img_raw_skeleton = np.genfromtxt(
        #     "/home/fei/catheter_reconstruction_ws/saved_images_calibration_case1/seg_images_calibration_case1/seg_left_recif_1_skeleton.csv",
        #     delimiter=',')

        # img_path = "../exp_data_dvrk/seg_video5/seg_left_recif_0.png"
        downscale = 1.0
        # This doesn't make that big of a difference on synthetic images
        gaussian_blur_kern_size = 5
        dilate_iterations = 1

        # image size
        self.res_width = 640
        self.res_height = 480
        self.show_every_so_many_samples = 10
        self.R = 0.0013

        # camera E parameters
        cam_RT_H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        invert_y = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.cam_RT_H = torch.matmul(invert_y, cam_RT_H)

        # camera I parameters
        self.cam_K = torch.tensor([[883.00220751, 0.0, 320.0], [0.0, 883.00220751, 240.0], [0, 0, 1.0]])
        self.cam_K = self.cam_K / downscale
        self.cam_K[-1, -1] = 1

        self.Fourier_order_N = 1

        raw_img_rgb = cv2.imread(img_path)
        # self.cam_distCoeffs = torch.tensor([-4.0444238705587998e-01, 5.8161897902897197e-01, -4.9797819387316098e-03, 2.3217574337593299e-03, -2.1547479006608700e-01])
        # raw_img_rgb_undst = cv2.undistort(raw_img_rgb, self.cam_K.detach().numpy(), self.cam_distCoeffs.detach().numpy())
        self.raw_img_rgb = cv2.resize(raw_img_rgb,
                                      (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
        self.raw_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

        # self.blur_raw_img = cv2.GaussianBlur(self.raw_img, (gaussian_blur_kern_size, gaussian_blur_kern_size), 0)
        # edges_img = canny(self.blur_raw_img, 2, 1, 100)
        # self.edges_img = cv2.dilate(edges_img.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=dilate_iterations)

        # self.optimizer = torch.optim.SGD([self.para], lr=1e-5)
        self.optimizer = torch.optim.Adam([self.para], lr=1e-3)

        # self.optimal_R_EMraw2Cam = torch.tensor([[0.40533652, -0.91020415, 0.08503356],
        #                                          [0.86140179, 0.41142924, 0.29784715],
        #                                          [-0.30608701, -0.04748027, 0.95081879]])
        # self.optimal_t_EMraw2Cam = torch.tensor([[-0.120146], [-0.20414568], [0.22804266]])
        self.GD_Iteration = 0
        self.loss = None

        self.saved_opt_history = np.zeros((1, self.para.shape[0] + 1))

        ## get raw image skeleton
        self.getContourSamples()

        ## get ground truth 3D bezier curve
        self.pos_bezier_3D_gt = self.getAnyBezierCurve(self.para_gt, self.P0_gt)
        self.pos_bezier_3D_init = self.getAnyBezierCurve(para_init, self.P0_gt)
'''
    
    def __init__(self): 
        super().__init__()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.epsilon = 1e-8

        # Number of samples to take along Bezier Curve
        # self.num_samples = 30
        self.num_samples = 30
        # Number of samples to take on INSIDE of each circle
        # self.samples_per_circle = 20
        self.samples_per_circle = 1
        # Number of samples to take on OUTSIDE border of each circle
        # self.bezier_surface_resolution = 20
        self.bezier_surface_resolution = 1
        self.bezier_circle_angle_increment = (2 * math.pi) / self.bezier_surface_resolution

        self.cylinder_mesh_points1 = torch.zeros(self.num_samples, self.samples_per_circle, 3)
        self.cylinder_surface_points1 = torch.zeros(self.num_samples, self.bezier_surface_resolution, 3)

        self.cylinder_mesh_points2 = torch.zeros(self.num_samples, self.samples_per_circle, 3)
        self.cylinder_surface_points2 = torch.zeros(self.num_samples, self.bezier_surface_resolution, 3)



        self.radius = 0.0015
        # self.radius = 2
        # self.radius = 0.003

        self.setCameraParams(camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y,
                             camera_settings.image_size_x, camera_settings.image_size_y, camera_settings.extrinsics,
                             camera_settings.intrinsics)
        
        # Used to generate same set of random numbers each time (mainly used for getting random points in circles)
        torch.manual_seed(0)

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
        # self.cam_RT_H = torch.as_tensor(camera_extrinsics).float()
        self.cam_K = torch.as_tensor(camera_intrinsics)

        # camera E parameters
        cam_RT_H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        invert_y = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.cam_RT_H = torch.matmul(invert_y, cam_RT_H)

    def getBezierCurve(self, control_pts):
        """ takes control points as input, calculates positions and 
        derivatives of points along the Bezier curve, converts the 
        coordinates to the camera frame, and stores the results in 
        self.pos_bezier_cam and self.der_bezier_cam respectively. 

        control_pts -- 
        """

        self.num_samples = 200
        P1 = control_pts[0, :]
        P1p = control_pts[1, :]
        P2 = control_pts[2, :]
        P2p = control_pts[3, :]

        sample_list = torch.linspace(0, 1, self.num_samples)

        # Get positions and normals from samples along bezier curve
        pos_bezier = torch.zeros(self.num_samples, 3)
        der_bezier = torch.zeros(self.num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P1 + 3 * s * (1 - s)**2 * \
                P1p + 3 * (1 - s) * s**2 * P2p + s**3 * P2
            der_bezier[i, :] = -(1 - s)**2 * P1 + ((1 - s)**2 - 2 * s * (1 - s)) * P1p + (-s**2 + 2 *
                                                                                          (1 - s) * s) * P2p + s**2 * P2

        # Convert positions and normals to camera frame
        self.pos_bezier_3D = pos_bezier
        pos_bezier_H = torch.cat((pos_bezier, torch.ones(self.num_samples, 1)), dim=1)

        pos_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(pos_bezier_H, 0, 1)), 0, 1)
        self.pos_bezier_cam = pos_bezier_cam_H[1:, :-1]

        # print(pos_bezier)
        # pos_bezier.register_hook(print)
        # P1.register_hook(print)
        # P1p.register_hook(print)
        # P2.register_hook(print)
        # P2p.register_hook(print)

        der_bezier_H = torch.cat((der_bezier, torch.zeros((self.num_samples, 1))), dim=1)
        der_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[1:, :], 0, 1)), 0,
                                           1)
        self.der_bezier_cam = der_bezier_cam_H[:, :-1]

        # pdb.set_trace() 

    def getAnyBezierCurve(self, para, P0):

        num_samples = 200
        sample_list = torch.linspace(0, 1, num_samples)

        P1 = torch.tensor([0.02, 0.002, 0.0])
        PC = torch.tensor([para[0], para[1], para[2]])
        P2 = torch.tensor([para[3], para[4], para[5]])
        P1p = 2 / 3 * PC + 1 / 3 * P1
        P2p = 2 / 3 * PC + 1 / 3 * P2

        # Get positions and normals from samples along bezier curve
        pos_bezier = torch.zeros(num_samples, 3)
        der_bezier = torch.zeros(num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P1 + 3 * s * (1 - s)**2 * \
                P1p + 3 * (1 - s) * s**2 * P2p + s**3 * P2

        return pos_bezier

    '''
    def getProjPointCam(self, p, cam_K):
        # p is of size R^(Nx3)
        if p.shape == (3, ):
            p = torch.unsqueeze(p, dim=0)

        divide_z = torch.div(torch.transpose(p[:, :-1], 0, 1), p[:, -1])

        # print(p[:, -1].transpose(), '\n', '------------')

        divide_z = torch.cat((divide_z, torch.ones(1, p.shape[0])), dim=0)

        # print(torch.matmul(cam_K, divide_z)[:-1, :])
        # pdb.set_trace()
        return torch.transpose(torch.matmul(cam_K, divide_z)[:-1, :], 0, 1)
    '''
    
    def isPointInImage(self, p_proj, width, height):
        if torch.all(torch.isnan(p_proj)):
            # print('NaN')
            return False
        if p_proj[0] < 0 or p_proj[1] < 0 or p_proj[0] > width or p_proj[1] > height:
            # print('Out')
            return False

        return True

###################################################################################################
###################################################################################################
###################################################################################################

# Helper functions for plotting using matplotlib

    def set_axes_equal(self, ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        self._set_axes_radius(ax, origin, radius)

    def _set_axes_radius(self, ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])

###################################################################################################
###################################################################################################
###################################################################################################

# Helper functions for obtaining the tangent, normal and binormal vectors of a bezier curve

    def getBezierTNB(self, bezier_pos, bezier_der, bezier_snd_der):

        bezier_der_n = torch.linalg.norm(bezier_der, ord=2, dim=1)
        # self.bezier_tangent = bezier_der / torch.unsqueeze(bezier_der_n, dim=1)

        bezier_normal_numerator = torch.linalg.cross(bezier_der, torch.linalg.cross(bezier_snd_der, bezier_der))
        bezier_normal_numerator_n = torch.mul(
            bezier_der_n, torch.linalg.norm(torch.linalg.cross(bezier_snd_der, bezier_der), ord=2, dim=1))

        bezier_normal = bezier_normal_numerator / (torch.unsqueeze(bezier_normal_numerator_n, dim=1) + self.epsilon)

        bezier_binormal_numerator = torch.linalg.cross(bezier_der, bezier_snd_der)
        bezier_binormal_numerator_n = torch.linalg.norm(bezier_binormal_numerator, ord=2, dim=1)

        bezier_binormal = bezier_binormal_numerator / (torch.unsqueeze(bezier_binormal_numerator_n, dim=1) + self.epsilon)

        print("bezier_normal: " + str(bezier_normal))
        print("bezier_binormal" + str(bezier_binormal))

        # pdb.set_trace()

        assert not torch.any(torch.isnan(bezier_normal))
        assert not torch.any(torch.isnan(bezier_binormal))

    def getBezierNormal(self, bezier_der, bezier_snd_der): 
        bezier_der_n = torch.linalg.norm(bezier_der, ord=2, dim=1)
        # self.bezier_tangent = bezier_der / torch.unsqueeze(bezier_der_n, dim=1)

        bezier_normal_numerator = torch.linalg.cross(bezier_der, torch.linalg.cross(bezier_snd_der, bezier_der))
        bezier_normal_numerator_n = torch.mul(
            bezier_der_n, torch.linalg.norm(torch.linalg.cross(bezier_snd_der, bezier_der), ord=2, dim=1))

        bezier_normal = bezier_normal_numerator / (torch.unsqueeze(bezier_normal_numerator_n, dim=1) + self.epsilon)

        # print("bezier_normal: " + str(bezier_normal))

        # Throw an error if there are an NaN values in bezier_normal
        assert not torch.any(torch.isnan(bezier_normal))

        return bezier_normal
    
    def getBezierBinormal(self, bezier_der, bezier_snd_der): 
        bezier_binormal_numerator = torch.linalg.cross(bezier_der, bezier_snd_der)
        bezier_binormal_numerator_n = torch.linalg.norm(bezier_binormal_numerator, ord=2, dim=1)

        bezier_binormal = bezier_binormal_numerator / (torch.unsqueeze(bezier_binormal_numerator_n, dim=1) + self.epsilon)

        # print("bezier_binormal" + str(bezier_binormal))

        # Throw an error if there are an NaN values in bezier_binormal
        assert not torch.any(torch.isnan(bezier_binormal))

        return bezier_binormal

# Helper functions for obtaining normalized and translated versions of vectors

    def getNormalizedVectors(self, set_of_vectors): 
        '''
        Method to get the normalized version of a set of vectors (of the shape: (num_samples, 3)). 
        Calculates the L2 norm (cartesian magnitude) of each vector and divides by it. 
        When vector norm is zero, dividing the vector by its norm will result in NaN values 
        because division by zero is mathematically undefined. To avoid NaN values, add a small epsilon 
        value to the denominator before performing the division. 
        This will prevent division by exactly zero and keep the vectors valid. 
        '''
        normalized_set_of_vectors = set_of_vectors / (torch.linalg.norm(set_of_vectors, ord=2, dim=0) + self.epsilon)
        return normalized_set_of_vectors

    def getTranslatedVectors(self, pos_bezier, set_of_vectors): 
        '''
        Method to get the translated version of a set of vectors (of the shape: (num_samples, 3)). 
        Adds respective point on Bezier curve to the vector (s.t. point is considered 'start' of translated vector). 
        '''
        translated_set_of_vectors = pos_bezier + set_of_vectors
        return translated_set_of_vectors


###################################################################################################
###################################################################################################
###################################################################################################

    # Functions for plotting the 3D Bezier curve (3d model vectors or 3d model cylinder) and 2D helpers

    def getRandCirclePoint(self, radius, center_point, normal_vec, binormal_vec): 
        '''
        Method to calculate random point on a circle in 3-dimensions. 

        Args: 
            radius: radius value of circle
            center_point (tensor): center point of circle; i.e., current point on Bezier curve
            normal_vec: normal vector at that point on Bezier curve
            binormal_vec: binormal vector at that point on Bezier curve
        '''
        rand_dist_from_center = radius * torch.sqrt(torch.rand(1))
        rand_angle = 2 * math.pi * torch.rand(1)

        rand_circle_point = center_point + rand_dist_from_center * (torch.cos(rand_angle)) * normal_vec + rand_dist_from_center * (torch.sin(rand_angle)) * binormal_vec

        return rand_circle_point

    def getCircleBorderPoint(self, radius, angle, center_point, normal_vec, binormal_vec):
        '''
        Method to calculate point on the border of a circle in 3-dimensions. 

        Args: 
            radius: radius value of circle
            center_point (tensor): center point of circle; i.e., current point on Bezier curve
            normal_vec: normal vector at that point on Bezier curve
            binormal_vec: binormal vector at that point on Bezier curve
        '''
        
        # Convert angle to a torch tensor to use torch's trig functions
        angle += torch.tensor(1)

        circle_border_point = center_point + radius * (torch.cos(angle)) * normal_vec + radius * (torch.sin(angle)) * binormal_vec

        return circle_border_point

    def plot3dPoints(self, show_vector_lines, plot_bezier_points, set_of_vectors=None): 
        '''
        Method to plot Bezier vectors using MatPlotLib

        Args: 
            pos_bezier (Tensor): Points along Bezier curve
            set_of_vectors (Tensor): Vectors (tangent, normal, binormal) along Bezier curve
            show_vector_lines (boolean): Boolean to show vector line between vector initial and terminal points
            plot_bezier_points (boolean): Boolean to plot points along Bezier curve
        '''
        # print("INSIDE pos_bezier: " + str(pos_bezier))
        # print("INSIDE set_of_vectors: " + str(set_of_vectors))

        # Only plot points along Bezier curve. No vector lines
        if plot_bezier_points is True and set_of_vectors is None and show_vector_lines is False: 
            for point in self.pos_bezier: 
                self.ax.scatter(point[0], point[1], point[2])

        # Only plot vector points. No vector lines
        elif plot_bezier_points is False and set_of_vectors is not None and show_vector_lines is False: 
            vec_normalized = self.getNormalizedVectors(set_of_vectors)
            vec_normalized_translated = self.getTranslatedVectors(self.pos_bezier, vec_normalized)

            for vec in vec_normalized_translated: 
                self.ax.scatter(vec[0], vec[1], vec[2])

        # Only plot vectors points. Show vector lines
        elif plot_bezier_points is False and set_of_vectors is not None and show_vector_lines is True:
            vec_normalized = self.getNormalizedVectors(set_of_vectors)
            vec_normalized_translated = self.getTranslatedVectors(self.pos_bezier, vec_normalized)
            
            for pos_vec, vec in zip(self.pos_bezier, vec_normalized_translated): 
                self.ax.scatter(vec[0], vec[1], vec[2])
                self.ax.plot([pos_vec[0], vec[0]], [pos_vec[1], vec[1]], [pos_vec[2], vec[2]])

        # Plot points along Bezier curve and vectors points. No vector lines
        elif plot_bezier_points is True and set_of_vectors is not None and show_vector_lines is False: 
            vec_normalized = self.getNormalizedVectors(set_of_vectors)
            vec_normalized_translated = self.getTranslatedVectors(self.pos_bezier, vec_normalized)

            for point, vec in zip(self.pos_bezier, vec_normalized_translated): 
                self.ax.scatter(point[0], point[1], point[2])
                self.ax.scatter(vec[0], vec[1], vec[2])

        # Plot points along Bezier curve and vectors points. Show vector lines
        elif plot_bezier_points is True and  set_of_vectors is not None and show_vector_lines is True: 
            vec_normalized = self.getNormalizedVectors(set_of_vectors)
            vec_normalized_translated = self.getTranslatedVectors(self.pos_bezier, vec_normalized)

            for point, vec in zip(self.pos_bezier, vec_normalized_translated): 
                self.ax.scatter(point[0], point[1], point[2])
                self.ax.scatter(vec[0], vec[1], vec[2])
                self.ax.plot([pos_vec[0], vec[0]], [pos_vec[1], vec[1]], [pos_vec[2], vec[2]])


    def run3dPlot(self): 

        self.ax.set_box_aspect([2,2,2]) 
        self.set_axes_equal(self.ax)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.fig.suptitle('Bézier Curve TNB Frames')

        plt.show()

    def plot3dBezierCylinder(self): 
        # Get Cylinder mesh points
        for i, (pos_vec) in enumerate(self.pos_bezier): 
            for j in range(self.samples_per_circle + self.bezier_surface_resolution): 

                # Plot cylinder mesh points
                self.ax.scatter(pos_vec[0].detach().numpy() + self.cylinder_mesh_and_surface_points[i, j, 0].detach().numpy(), 
                                pos_vec[1].detach().numpy() + self.cylinder_mesh_and_surface_points[i, j, 1].detach().numpy(), 
                                pos_vec[2].detach().numpy() + self.cylinder_mesh_and_surface_points[i, j, 2].detach().numpy())

        # Set up axes for 3d plot
        self.ax.set_box_aspect([2,2,2]) 
        self.set_axes_equal(self.ax)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.fig.suptitle('Bézier Curve - Cross Sectional Circles')

        plt.show()

    def plotTwo3dBezierCylinders(self): 
        # PLOT Cylinder 1
        for i, (pos_vec) in enumerate(self.pos_bezier1): 
            for j in range(self.samples_per_circle + self.bezier_surface_resolution): 

                # Plot cylinder mesh points
                self.ax.scatter(pos_vec[0].detach().numpy() + self.cylinder_mesh_and_surface_points1[i, j, 0].detach().numpy(), 
                                pos_vec[1].detach().numpy() + self.cylinder_mesh_and_surface_points1[i, j, 1].detach().numpy(), 
                                pos_vec[2].detach().numpy() + self.cylinder_mesh_and_surface_points1[i, j, 2].detach().numpy(), 
                                c='r')
                
        # Plot Cylinder 2
        for i, (pos_vec) in enumerate(self.pos_bezier2): 
            for j in range(self.samples_per_circle + self.bezier_surface_resolution): 

                # Plot cylinder mesh points
                self.ax.scatter(pos_vec[0].detach().numpy() + self.cylinder_mesh_and_surface_points2[i, j, 0].detach().numpy(), 
                                pos_vec[1].detach().numpy() + self.cylinder_mesh_and_surface_points2[i, j, 1].detach().numpy(), 
                                pos_vec[2].detach().numpy() + self.cylinder_mesh_and_surface_points2[i, j, 2].detach().numpy()
                                , c='b')


        # Set up axes for 3d plot
        self.ax.set_box_aspect([2,2,2]) 
        self.set_axes_equal(self.ax)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.fig.suptitle('Prediction vs Ground Truth 3D Models: Iteration 100')

        plt.show()



###################################################################################################
###################################################################################################
###################################################################################################


    def getBezierCurveCylinder1(self, p_start, para_gt): 
        '''
        Method to obtain bezier curve position, tangents, normals, and binormals. 
        Calls helper methods to plot these vectors. 

        Args: 
            para_gt: Ground truth parameters for bezier curve. Extract bezier control points from this. 
                     i.e., para_gt[0:3] is starting point for bezier curve. 
            
            Deprecated: 
                p_start: Starting point for bezier curve (used to be fixed and not updated by training)
                control_pts (tensor of shape [4, 3]): contains the control points for the Bezier curve
        '''
        
        # Get control points from ground truth parameters
        P0 = p_start
        P1 = para_gt[0:3]
        P2 = para_gt[3:6]

        # p_start = para_gt[0:3]
        # p_mid = para_gt[3:6]
        # p_end = para_gt[6:9]
        # p_c2 = 4 / 3 * p_mid - 1 / 3 * p_start
        # p_c1 = 4 / 3 * p_mid - 1 / 3 * p_end

        # P0 = p_start
        # P1 = p_c1
        # P2 = p_c2

        # P0 = control_pts[0, :]
        # P1 = control_pts[1, :]
        # P2 = control_pts[2, :]
        # P3 = control_pts[3, :]

        sample_list = torch.linspace(0, 1, self.num_samples)

        # print("\n Sample list: " + str(sample_list))

        # Get positions and normals [NOTE: SHOULD be tangents?] from samples along bezier curve
        self.pos_bezier1 = torch.zeros(self.num_samples, 3)
        self.der_bezier1 = torch.zeros(self.num_samples, 3)
        self.double_der_bezier1 = torch.zeros(self.num_samples, 3)
        for i, s in enumerate(sample_list):
            # pos_bezier[i, :] = (1 - s)**3 * P0 + 3 * s * (1 - s)**2 * \
            #     P1 + 3 * (1 - s) * s**2 * P2 + s**3 * P3
            # der_bezier[i, :] = 3 * (1 - s)**2 * (P1 - P0) + 6 * (1 - s) * s * (P2 - P1) + 3 * s**2 * (P3 - P2)
            # double_der_bezier[i, :] = 6 * (1 - s) * (P2 - 2*P1 + P0) + 6 * (P3 - 2*P2 + P1) * s

            self.pos_bezier1[i, :] = (1 - s) ** 2 * P0 + 2 * (1 - s) * s * P1 + s ** 2 * P2
            self.der_bezier1[i, :] = 2 * (1 - s) * (P1 - P0) + 2 * s * (P2 - P1)
            self.double_der_bezier1[i, :] = 2 * (P2 - 2 * P1 +  P0)


        # Get normal and binormals at samples along bezier curve
        self.normal_bezier1 = torch.zeros(self.num_samples, 3)
        self.binormal_bezier1 = torch.zeros(self.num_samples, 3)
        self.normal_bezier1 = self.getBezierNormal(self.der_bezier1, self.double_der_bezier1)
        self.binormal_bezier1 = self.getBezierBinormal(self.der_bezier1, self.double_der_bezier1)     

        '''
        print("OG VECTORS")
        print("self.pos_bezier: " + str(pself.os_bezier))
        print("self.normal_bezier: " + str(self.normal_bezier))
        print("self.binormal_bezier: " + str(self.binormal_bezier))
        '''
        

        # Plot TNB frames for all samples along bezier curve
        '''
        # Plot points along Bezier curve
        self.plot3dPoints(False, True, self.pos_bezier)
        # Plot tangents along Bezier curve
        self.plot3dPoints(True, False, self.pos_bezier, self.der_bezier)
        # Plot normals along Bezier curve
        self.plot3dPoints(True, False, self.pos_bezier, self.normal_bezier)
        # Plot binormals along Bezier curve
        self.plot3dPoints(True, False, self.pos_bezier, self.binormal_bezier)
        '''
        

        # Get Cylinder mesh points
        for i, (pos_vec, normal_vec, binormal_vec) in enumerate(zip(self.pos_bezier1, self.normal_bezier1, self.binormal_bezier1)): 
            for j in range(self.samples_per_circle): 
                normal_vec_normalized = self.getNormalizedVectors(normal_vec)
                binormal_vec_normalized = self.getNormalizedVectors(binormal_vec)
                self.cylinder_mesh_points1[i, j, :] = self.getRandCirclePoint(self.radius, pos_vec, normal_vec_normalized, binormal_vec_normalized)
    
        # Get Cylinder surface points. Combine into cylinder mesh points. 
        for i, (pos_vec, normal_vec, binormal_vec) in enumerate(zip(self.pos_bezier1, self.normal_bezier1, self.binormal_bezier1)):
            for j in range(self.bezier_surface_resolution):
                normal_vec_normalized = self.getNormalizedVectors(normal_vec)
                binormal_vec_normalized = self.getNormalizedVectors(binormal_vec)
                self.cylinder_surface_points1[i, j, :] = self.getCircleBorderPoint(self.radius, j * self.bezier_circle_angle_increment, pos_vec, normal_vec_normalized, binormal_vec_normalized)

        # Stack self.cylinder_mesh_points and self.cylinder_surface_points on top of each other in dimension 1
        self.cylinder_mesh_and_surface_points1 = torch.cat((self.cylinder_mesh_points1, self.cylinder_surface_points1), dim=1)

        # print("self.cylinder_mesh_and_surface_points.shape: " + str(self.cylinder_mesh_and_surface_points.shape))
        # print("self.cylinder_mesh_and_surface_points: " + str(self.cylinder_mesh_and_surface_points))


    def getBezierCurveCylinder2(self, p_start, para_gt): 
        '''
        Method to obtain bezier curve position, tangents, normals, and binormals. 
        Calls helper methods to plot these vectors. 

        Args: 
            para_gt: Ground truth parameters for bezier curve. Extract bezier control points from this. 
                     i.e., para_gt[0:3] is starting point for bezier curve. 
            
            Deprecated: 
                p_start: Starting point for bezier curve (used to be fixed and not updated by training)
                control_pts (tensor of shape [4, 3]): contains the control points for the Bezier curve
        '''
        
        # Get control points from ground truth parameters
        P0 = p_start
        P1 = para_gt[0:3]
        P2 = para_gt[3:6]

        # p_start = para_gt[0:3]
        # p_mid = para_gt[3:6]
        # p_end = para_gt[6:9]
        # p_c2 = 4 / 3 * p_mid - 1 / 3 * p_start
        # p_c1 = 4 / 3 * p_mid - 1 / 3 * p_end

        # P0 = p_start
        # P1 = p_c1
        # P2 = p_c2

        # P0 = control_pts[0, :]
        # P1 = control_pts[1, :]
        # P2 = control_pts[2, :]
        # P3 = control_pts[3, :]

        sample_list = torch.linspace(0, 1, self.num_samples)

        # print("\n Sample list: " + str(sample_list))

        # Get positions and normals [NOTE: SHOULD be tangents?] from samples along bezier curve
        self.pos_bezier2 = torch.zeros(self.num_samples, 3)
        self.der_bezier2 = torch.zeros(self.num_samples, 3)
        self.double_der_bezier2 = torch.zeros(self.num_samples, 3)
        for i, s in enumerate(sample_list):
            # pos_bezier[i, :] = (1 - s)**3 * P0 + 3 * s * (1 - s)**2 * \
            #     P1 + 3 * (1 - s) * s**2 * P2 + s**3 * P3
            # der_bezier[i, :] = 3 * (1 - s)**2 * (P1 - P0) + 6 * (1 - s) * s * (P2 - P1) + 3 * s**2 * (P3 - P2)
            # double_der_bezier[i, :] = 6 * (1 - s) * (P2 - 2*P1 + P0) + 6 * (P3 - 2*P2 + P1) * s

            self.pos_bezier2[i, :] = (1 - s) ** 2 * P0 + 2 * (1 - s) * s * P1 + s ** 2 * P2
            self.der_bezier2[i, :] = 2 * (1 - s) * (P1 - P0) + 2 * s * (P2 - P1)
            self.double_der_bezier2[i, :] = 2 * (P2 - 2 * P1 +  P0)


        # Get normal and binormals at samples along bezier curve
        self.normal_bezier2 = torch.zeros(self.num_samples, 3)
        self.binormal_bezier2 = torch.zeros(self.num_samples, 3)
        self.normal_bezier2 = self.getBezierNormal(self.der_bezier2, self.double_der_bezier2)
        self.binormal_bezier2 = self.getBezierBinormal(self.der_bezier2, self.double_der_bezier2)     

        '''
        print("OG VECTORS")
        print("self.pos_bezier: " + str(pself.os_bezier))
        print("self.normal_bezier: " + str(self.normal_bezier))
        print("self.binormal_bezier: " + str(self.binormal_bezier))
        '''
        

        # Plot TNB frames for all samples along bezier curve
        '''
        # Plot points along Bezier curve
        self.plot3dPoints(False, True, self.pos_bezier)
        # Plot tangents along Bezier curve
        self.plot3dPoints(True, False, self.pos_bezier, self.der_bezier)
        # Plot normals along Bezier curve
        self.plot3dPoints(True, False, self.pos_bezier, self.normal_bezier)
        # Plot binormals along Bezier curve
        self.plot3dPoints(True, False, self.pos_bezier, self.binormal_bezier)
        '''
        

        # Get Cylinder mesh points
        for i, (pos_vec, normal_vec, binormal_vec) in enumerate(zip(self.pos_bezier2, self.normal_bezier2, self.binormal_bezier2)): 
            for j in range(self.samples_per_circle): 
                normal_vec_normalized = self.getNormalizedVectors(normal_vec)
                binormal_vec_normalized = self.getNormalizedVectors(binormal_vec)
                self.cylinder_mesh_points2[i, j, :] = self.getRandCirclePoint(self.radius, pos_vec, normal_vec_normalized, binormal_vec_normalized)
    
        # Get Cylinder surface points. Combine into cylinder mesh points. 
        for i, (pos_vec, normal_vec, binormal_vec) in enumerate(zip(self.pos_bezier2, self.normal_bezier2, self.binormal_bezier2)):
            for j in range(self.bezier_surface_resolution):
                normal_vec_normalized = self.getNormalizedVectors(normal_vec)
                binormal_vec_normalized = self.getNormalizedVectors(binormal_vec)
                self.cylinder_surface_points2[i, j, :] = self.getCircleBorderPoint(self.radius, j * self.bezier_circle_angle_increment, pos_vec, normal_vec_normalized, binormal_vec_normalized)

        # Stack self.cylinder_mesh_points and self.cylinder_surface_points on top of each other in dimension 1
        self.cylinder_mesh_and_surface_points2 = torch.cat((self.cylinder_mesh_points2, self.cylinder_surface_points2), dim=1)

        # print("self.cylinder_mesh_and_surface_points.shape: " + str(self.cylinder_mesh_and_surface_points.shape))
        # print("self.cylinder_mesh_and_surface_points: " + str(self.cylinder_mesh_and_surface_points))


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

if __name__ == '__main__': 

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
    start_gt = torch.tensor([0.02, 0.008, 0.054])

    para_final = torch.tensor([ 0.0167, 0.0131, 0.0984, 0.0059, -0.0449, 0.2355])
    para_gt = torch.tensor([0.02003904, 0.0016096, 0.13205799, 0.00489567, -0.03695673, 0.196168896])

    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/diff_render_1'
    img_save_path = case_naming + '.png'


    ###========================================================
    ### 3) SETTING UP BEZIER CURVE CONSTRUCTION
    ###========================================================
    build_two_bezier_curves = ConstructionBezier()


    ###========================================================
    ### 4) RUNNING BEZIER CURVE CONSTRUCTION
    ###========================================================
    # Generate the Bezier curve cylinder mesh points
    build_two_bezier_curves.getBezierCurveCylinder1(start_gt, para_final)
    build_two_bezier_curves.getBezierCurveCylinder2(start_gt, para_gt)

    # Plot 3D Bezier Cylinder mesh points
    build_two_bezier_curves.plotTwo3dBezierCylinders()