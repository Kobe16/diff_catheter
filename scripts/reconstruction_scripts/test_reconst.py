import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.figsize'] = [1280, 800]
# mpl.rcParams['figure.dpi'] = 300

import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.morphology import skeletonize

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from random import random

import torch
import torch.nn.functional as F
import shutil
import os
import pdb
import argparse

class reconstructCurve():
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

    def isPointInImage(self, p_proj, width, height):
        if torch.all(torch.isnan(p_proj)):
            # print('NaN')
            return False
        if p_proj[0] < 0 or p_proj[1] < 0 or p_proj[0] > width or p_proj[1] > height:
            # print('Out')
            return False

        return True


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


    def getBezierTNB(self, bezier_pos, bezier_der, bezier_snd_der):

        bezier_der_n = torch.linalg.norm(bezier_der, ord=2, dim=1)
        # self.bezier_tangent = bezier_der / torch.unsqueeze(bezier_der_n, dim=1)

        bezier_normal_numerator = torch.linalg.cross(bezier_der, torch.linalg.cross(bezier_snd_der, bezier_der))
        bezier_normal_numerator_n = torch.mul(
            bezier_der_n, torch.linalg.norm(torch.linalg.cross(bezier_snd_der, bezier_der), ord=2, dim=1))

        bezier_normal = bezier_normal_numerator / torch.unsqueeze(bezier_normal_numerator_n, dim=1)

        bezier_binormal_numerator = torch.linalg.cross(bezier_der, bezier_snd_der)
        bezier_binormal_numerator_n = torch.linalg.norm(bezier_binormal_numerator, ord=2, dim=1)

        bezier_binormal = bezier_binormal_numerator / torch.unsqueeze(bezier_binormal_numerator_n, dim=1)

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

        bezier_normal = bezier_normal_numerator / torch.unsqueeze(bezier_normal_numerator_n, dim=1)

        # print("bezier_normal: " + str(bezier_normal))

        # Throw an error if there are an NaN values in bezier_normal
        assert not torch.any(torch.isnan(bezier_normal))

        return bezier_normal
    
    def getBezierBinormal(self, bezier_der, bezier_snd_der): 
        bezier_binormal_numerator = torch.linalg.cross(bezier_der, bezier_snd_der)
        bezier_binormal_numerator_n = torch.linalg.norm(bezier_binormal_numerator, ord=2, dim=1)

        bezier_binormal = bezier_binormal_numerator / torch.unsqueeze(bezier_binormal_numerator_n, dim=1)

        # print("bezier_binormal" + str(bezier_binormal))

        # Throw an error if there are an NaN values in bezier_binormal
        assert not torch.any(torch.isnan(bezier_binormal))

        return bezier_binormal


    def getNormalizedVectors(self, set_of_vectors): 
        '''
        Method to get the normalized version of a set of vectors (of the shape: (num_samples, 3)). 
        Calculates the L2 norm (cartesian magnitude) of each vector and divides by it. 
        '''
        normalized_set_of_vectors = set_of_vectors / torch.linalg.norm(set_of_vectors, ord=2, dim=0)
        return normalized_set_of_vectors

    def getTranslatedVectors(self, pos_bezier, set_of_vectors): 
        '''
        Method to get the translated version of a set of vectors (of the shape: (num_samples, 3)). 
        Adds respective point on Bezier curve to the vector (s.t. point is considered 'start' of translated vector). 
        '''
        translated_set_of_vectors = pos_bezier + set_of_vectors
        return translated_set_of_vectors


    def getRandCirclePoint(self, radius, center_point, normal_vec, binormal_vec): 
        '''
        Method to calculate random point on a circle in 3-dimensions. 

        Args: 
            radius: radius value of circle
            center_point (tensor): center point of circle; i.e., current point on Bezier curve
            normal_vec: normal vector at that point on Bezier curve
            binormal_vec: binormal vector at that point on Bezier curve
        '''
        rand_dist_from_center = radius * math.sqrt(random())
        rand_angle = 2 * math.pi * random()

        rand_circle_point = center_point + rand_dist_from_center * (math.cos(rand_angle)) * normal_vec + rand_dist_from_center * (math.sin(rand_angle)) * binormal_vec

        return rand_circle_point


    def getBezierCurveCylinder(self, control_pts, radius): 
        
        self.num_samples = 15
        self.points_per_circle = 15
        self.cylinder_mesh_points = torch.zeros(self.num_samples, self.points_per_circle, 3)
        
        #self.num_samples = 200
        P0 = control_pts[0, :]
        P1 = control_pts[1, :]
        P2 = control_pts[2, :]
        P3 = control_pts[3, :]

        sample_list = torch.linspace(0, 1, self.num_samples)

        # print("\n Sample list: " + str(sample_list))

        # Get positions and normals [NOTE: SHOULD be tangents?] from samples along bezier curve
        pos_bezier = torch.zeros(self.num_samples, 3)
        der_bezier = torch.zeros(self.num_samples, 3)
        double_der_bezier = torch.zeros(self.num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P0 + 3 * s * (1 - s)**2 * \
                P1 + 3 * (1 - s) * s**2 * P2 + s**3 * P3
            der_bezier[i, :] = 3 * (1 - s)**2 * (P1 - P0) + 6 * (1 - s) * s * (P2 - P1) + 3 * s**2 * (P3 - P2)
            double_der_bezier[i, :] = 6 * (1 - s) * (P2 - 2*P1 + P0) + 6 * (P3 - 2*P2 + P1) * s

        # Get normal and binormals at samples along bezier curve
        normal_bezier = torch.zeros(self.num_samples, 3)
        binormal_bezier = torch.zeros(self.num_samples, 3)
        normal_bezier = self.getBezierNormal(der_bezier, double_der_bezier)
        binormal_bezier = self.getBezierBinormal(der_bezier, double_der_bezier)     

        # print("pos_bezier: " + str(pos_bezier))
        # print("normal_bezier: " + str(normal_bezier))
        # print("binormal_bezier: " + str(binormal_bezier))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot TNB frames for all samples along bezier curve
        for pos_vec, tan_vec, normal_vec, binormal_vec  in zip(pos_bezier, der_bezier, normal_bezier, binormal_bezier): 

            # A set of vectors are mutually orthogonal if every pair of vectors is orthogonal (DP = 0)
            # print("tan . normal = " + str(torch.dot(tan_vec, normal_vec)))
            # print("tan . binormal = " + str(torch.dot(tan_vec, binormal_vec)))
            # print("normal . binormal = " + str(torch.dot(normal_vec, binormal_vec)))
            # print('\n')

            tan_vec_normalized = self.getNormalizedVectors(tan_vec)
            normal_vec__normalized = self.getNormalizedVectors(normal_vec)
            binormal_vec_normalized = self.getNormalizedVectors(binormal_vec)

            # print("tan_vec_normalized: " + str(tan_vec_normalized))
            # print("normal_vec__normalized" + str(normal_vec__normalized))
            # print("binormal_vec_normalized" + str(binormal_vec_normalized))
            
            # Points along Bezier curve
            # ax.scatter(pos_vec[0], pos_vec[1], pos_vec[2])

            # Tangents along Bezier curve
            # ax.scatter(pos_vec[0] + tan_vec_normalized[0], pos_vec[1] + tan_vec_normalized[1], pos_vec[2] + tan_vec_normalized[2])
            # ax.plot([pos_vec[0], pos_vec[0]+ tan_vec_normalized[0]], [pos_vec[1], pos_vec[1]+ tan_vec_normalized[1]], [pos_vec[2], pos_vec[2]+ tan_vec_normalized[2]])

            # Normals along Bezier curve
            # ax.scatter(pos_vec[0] + normal_vec__normalized[0], pos_vec[1] + normal_vec__normalized[1], pos_vec[2] + normal_vec__normalized[2])
            # ax.plot([pos_vec[0], pos_vec[0]+ normal_vec__normalized[0]], [pos_vec[1], pos_vec[1]+ normal_vec__normalized[1]], [pos_vec[2], pos_vec[2]+ normal_vec__normalized[2]])

            # Binormals along Bezier curve
            # ax.scatter(pos_vec[0] + binormal_vec_normalized[0], pos_vec[1] + binormal_vec_normalized[1], pos_vec[2] + binormal_vec_normalized[2])
            # ax.plot([pos_vec[0], pos_vec[0]+ binormal_vec_normalized[0]], [pos_vec[1], pos_vec[1]+ binormal_vec_normalized[1]], [pos_vec[2], pos_vec[2]+ binormal_vec_normalized[2]])
            
        # Plot Bezier curve points
        # for point in pos_bezier: 
        #     ax.scatter(point[0], point[1], point[2])

    
        # Plot Bezier curve points & tangent vectors. 
        # USE TRANSLATED DERIVATIVE VECTORS such that they start at their corresponding point
        # for point, tan_vector in zip(pos_bezier, trans_unit_der_bezier): 
        #     ax.scatter(point[0], point[1], point[2])
        #     ax.scatter(tan_vector[0], tan_vector[1], tan_vector[2])
        #     ax.plot([point[0], tan_vector[0]], [point[1], tan_vector[1]], [point[2], tan_vector[2]])


        # Plot Bezier curve points & normal vectors.
        # for point, norm_vector in zip(pos_bezier, normal_bezier): 
        #     ax.scatter(point[0], point[1], point[2])
        #     ax.scatter(norm_vector[0], norm_vector[1], norm_vector[2])
        #     ax.plot([point[0], norm_vector[0]], [point[1], norm_vector[1]], [point[2], norm_vector[2]])
        

        for i, (pos_vec, normal_vec, binormal_vec) in enumerate(zip(pos_bezier, normal_bezier, binormal_bezier)): 
            for j in range(self.points_per_circle): 
                normal_vec_normalized = self.getNormalizedVectors(normal_vec)
                binormal_vec_normalized = self.getNormalizedVectors(binormal_vec)
                self.cylinder_mesh_points[i, j, :] = self.getRandCirclePoint(radius, pos_vec, normal_vec_normalized, binormal_vec_normalized)
    
                # Plot cylinder mesh points
                ax.scatter(pos_vec[0] + self.cylinder_mesh_points[i, j, 0], pos_vec[1] + self.cylinder_mesh_points[i, j, 1], pos_vec[2] + self.cylinder_mesh_points[i, j, 2])


        ax.set_box_aspect([1,1,1]) 
        self.set_axes_equal(ax)
        plt.show()




a = reconstructCurve()
test_control_points1 = torch.tensor([[0., 0., 0.], 
                                    [0., 1., 2.5], 
                                    [0., 2., 0.5], 
                                    [0., 3., 3.]])

test_control_points2 = torch.tensor([[0., 0., 0.], 
                                    [2., 0.5, 2.5], 
                                    [-2., 2., 0.5], 
                                    [0., 3., 3.]])

test_control_points3 = torch.tensor([[0., 0., 0.], 
                                    [20., 5., 25.], 
                                    [-20., 20., 5.], 
                                    [0., 30., 30.]])

test_control_points4 = torch.tensor([[0., 0., 0.], 
                                    [5., 5., 5.], 
                                    [10., 10., 10.], 
                                    [15., 15., 15.]])

test_control_points5 = torch.tensor([[[0., 0., 0.], 
                                    [5., 5., 5.], 
                                    [10., 10., 10.], 
                                    [15., 15., 15.], 
                                    [15., 15., 15.]]])


a.getBezierCurveCylinder(test_control_points3, 10)
