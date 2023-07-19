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

    def getFrenetFrame(self, der_vector, double_der_vector): 
        ''' Helper function for getBezierCurveCylinder.
        All vectors in this method have shape of (1) and should be normalized 
        using L2 normalization
        '''

        normalized_a = F.normalize(der_vector, p=2.0, dim=0)
        # print("\nNormalized_a: " + str(normalized_a))
        normalized_b = F.normalize(normalized_a + double_der_vector, p=2.0, dim=0)
        # print("\nNormalized_b: " + str(normalized_b))
        rotational_axis_vector = F.normalize(torch.cross(normalized_a, normalized_b), p=2.0, dim=0)
        # print("\nrotational_axis_vector: " + str(rotational_axis_vector))
        normal_vector = F.normalize(torch.cross(rotational_axis_vector, normalized_a, dim=0), p=2.0, dim=0)
        # print("\nnormal_vector: " + str(normal_vector))

        return normal_vector
    
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


    def getRotatedVector(self, rot_angle, rot_axis_vector, rot_vector): 
        '''Helper function for getBezierCurveCylinder.
        Rotates a vector around an axis by a certain angle in 3D space. 

        rot_angle -- angle by which to rotate by. MUST BE RADIANS
        rot_axis_vector -- axis to rotate around. Will be the 3D tangent vector at the 
                           point on the Bezier curve. MUST BE A UNIT VECTOR
        rot_vector -- vector to rotate. Will be the 3D normal vector at the point on
                      the Bezier curve
        '''

        ux, uy, uz = rot_axis_vector
        # print("ux: " + str(ux))
        # print("uy: " + str(uy))
        # print("uz: " + str(uz))
        # print("Should be 1: " + str(ux**2 + uy**2 + uz**2))
        cos_theta = math.cos(rot_angle)
        sin_theta = math.sin(rot_angle)
        one_minus_cos_theta = 1 - cos_theta

        R = torch.tensor([
            [cos_theta + ux**2 * one_minus_cos_theta, ux * uy * one_minus_cos_theta - uz * sin_theta, ux * uz * one_minus_cos_theta + uy * sin_theta],
            [uy * ux * one_minus_cos_theta + uz * sin_theta, cos_theta + uy**2 * one_minus_cos_theta, uy * uz * one_minus_cos_theta - ux * sin_theta],
            [uz * ux * one_minus_cos_theta - uy * sin_theta, uz * uy * one_minus_cos_theta + ux * sin_theta, cos_theta + uz**2 * one_minus_cos_theta]
        ])
        # R = torch.tensor([[1., 0., 0.], 
        #                   [0., 1., 0.], 
        #                   [0., 0., 1.]])

        rotated_vector = torch.matmul(R, rot_vector)

        return rotated_vector
    
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


    def getBezierCurveCylinder(self, control_pts, radius): 
        
        self.num_samples = 20
        #self.num_samples = 200
        P0 = control_pts[0, :]
        P1 = control_pts[1, :]
        P2 = control_pts[2, :]
        P3 = control_pts[3, :]
        origin = [0, 0, 0]
        num_samples_on_circ = 8
        rotation_angle = math.radians(360 / num_samples_on_circ)

        # print(P0)
        # print(P1)
        # print(P2)
        # print(P3)

        sample_list = torch.linspace(0, 1, self.num_samples)

        # print("\n Sample list: ")
        # print(sample_list)

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

        # Normalize untrans_der_vector so that they are unit vectors
        # Translate unit_untrans_der_vector and unit_untrans_doubles_der_vector such that they start at their corresponding point
        # MUST NORMALIZE FIRST (idk why...)
        trans_unit_der_bezier = torch.zeros(self.num_samples, 3)
        trans_unit_double_der_bezier = torch.zeros(self.num_samples, 3)
        for i, (point, untrans_der_vector, untrans_double_der_vector) in enumerate(zip(pos_bezier, der_bezier, double_der_bezier)): 
            unit_untrans_der_vector = F.normalize(untrans_der_vector, p=2.0, dim=0)
            unit_untrans_doubles_der_vector = F.normalize(untrans_double_der_vector, p=2.0, dim=0)

            trans_unit_der_bezier[i, 0] = point[0] + unit_untrans_der_vector[0]
            trans_unit_der_bezier[i, 1] = point[1] + unit_untrans_der_vector[1]
            trans_unit_der_bezier[i, 2] = point[2] + unit_untrans_der_vector[2]

            trans_unit_double_der_bezier[i, 0] = point[0] + unit_untrans_doubles_der_vector[0]
            trans_unit_double_der_bezier[i, 1] = point[1] + unit_untrans_doubles_der_vector[1]
            trans_unit_double_der_bezier[i, 2] = point[2] + unit_untrans_doubles_der_vector[2]

            # trans_unit_der_bezier[i, :] = F.normalize(trans_unit_der_bezier[i, :], p=2.0, dim=0)
            # # print("Should be one: " + str(trans_unit_der_bezier[i, 0]**2 + trans_unit_der_bezier[i, 1]**2 + trans_unit_der_bezier[i, 2]**2))
            # trans_unit_double_der_bezier[i, :] = F.normalize(trans_unit_double_der_bezier[i, :], p=2.0, dim=0)


        # Get 3d normal unit vectors of points on bezier curve
        # normal_bezier = torch.zeros(self.num_samples, 3)
        # for i, (point, der_vector, double_der_vector) in enumerate(zip(pos_bezier, trans_unit_der_bezier, trans_unit_double_der_bezier)):
        #     normal_bezier[i, :] = self.getFrenetFrame(der_vector, double_der_vector)
            # Translate normal vectors such that they start at their corresponding point 
            # normal_bezier[i, 0] += point[0]
            # normal_bezier[i, 1] += point[1]
            # normal_bezier[i, 2] += point[2]
            # Don't need to normalize this again because getFrenetFrame returns a unit vector            
            

        # print("\n Positions of bezier points: ")
        # print(pos_bezier)
        # print("\n Derivatives of bezier points: ")
        # print(der_bezier)
        # print("\n TRANSLATED Derivatives of bezier points: ")
        # print(trans_unit_der_bezier)
        # print("\n Double Derivatives of bezier points: ")
        # print(double_der_bezier)
        # print("\n Normals of bezier points: ")
        # print(normal_bezier)

       
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot TNB frames for all samples along bezier curve
        for pos_vec, tan_vec, normal_vec, binormal_vec  in zip(pos_bezier, der_bezier, normal_bezier, binormal_bezier): 

            # A set of vectors are mutually orthogonal if every pair of vectors is orthogonal (DP = 0)
            print("tan . normal = " + str(torch.dot(tan_vec, normal_vec)))
            print("tan . binormal = " + str(torch.dot(tan_vec, binormal_vec)))
            print("normal . binormal = " + str(torch.dot(normal_vec, binormal_vec)))
            print('\n')

            tan_vec_normalized = tan_vec / torch.linalg.norm(tan_vec, ord=2, dim=0)
            normal_vec__normalized = normal_vec / torch.linalg.norm(normal_vec, ord=2, dim=0)
            binormal_vec_normalized = binormal_vec / torch.linalg.norm(binormal_vec, ord=2, dim=0)

            # print("tan . normal = " + str(torch.dot(pos_vec + tan_vec_normalized, pos_vec + normal_vec__normalized)))
            # print("tan . binormal = " + str(torch.dot(pos_vec + tan_vec_normalized, pos_vec + binormal_vec_normalized)))
            # print("normal . binormal = " + str(torch.dot(pos_vec + normal_vec__normalized, pos_vec + binormal_vec_normalized)))

            # print("tan_vec_normalized: " + str(tan_vec_normalized))
            # print("normal_vec__normalized" + str(normal_vec__normalized))
            # print("binormal_vec_normalized" + str(binormal_vec_normalized))
            
            # Points along Bezier curve
            ax.scatter(pos_vec[0], pos_vec[1], pos_vec[2])

            # Tangents along Bezier curve
            ax.scatter(pos_vec[0] + tan_vec_normalized[0], pos_vec[1] + tan_vec_normalized[1], pos_vec[2] + tan_vec_normalized[2])
            ax.plot([pos_vec[0], pos_vec[0]+ tan_vec_normalized[0]], [pos_vec[1], pos_vec[1]+ tan_vec_normalized[1]], [pos_vec[2], pos_vec[2]+ tan_vec_normalized[2]])

            # Normals along Bezier curve
            ax.scatter(pos_vec[0] + normal_vec__normalized[0], pos_vec[1] + normal_vec__normalized[1], pos_vec[2] + normal_vec__normalized[2])
            ax.plot([pos_vec[0], pos_vec[0]+ normal_vec__normalized[0]], [pos_vec[1], pos_vec[1]+ normal_vec__normalized[1]], [pos_vec[2], pos_vec[2]+ normal_vec__normalized[2]])

            # Binormals along Bezier curve
            ax.scatter(pos_vec[0] + binormal_vec_normalized[0], pos_vec[1] + binormal_vec_normalized[1], pos_vec[2] + binormal_vec_normalized[2])
            ax.plot([pos_vec[0], pos_vec[0]+ binormal_vec_normalized[0]], [pos_vec[1], pos_vec[1]+ binormal_vec_normalized[1]], [pos_vec[2], pos_vec[2]+ binormal_vec_normalized[2]])


            # ax.scatter(normal_vec[0], normal_vec[1], normal_vec[2])
            # ax.scatter(binormal_vec[0], binormal_vec[1], binormal_vec[2])
            

        # Create figure plotting bezier curve points
        # for point in pos_bezier: 
        #     ax.scatter(point[0], point[1], point[2])

        # Plot bezier curve points and the tangent vectors at those points. MUST USE TRANSLATED DERIVATIVE VECTORS such that they start at their 
        # corresponding point  -- NOT NEEDED NOW BECAUSE OF FOR LOOP UNDER THIS
        # for point, tan_vector in zip(pos_bezier, der_bezier): 
        #     normalized_tan_vector = F.normalize(tan_vector, p=2.0, dim=0)
        #     ax.scatter(point[0], point[1], point[2])
        #     ax.scatter(point[0] + normalized_tan_vector[0], point[1] + normalized_tan_vector[1], point[2] + normalized_tan_vector[2])
        #     ax.plot([point[0], point[0] + normalized_tan_vector[0]], [point[1], point[1] + normalized_tan_vector[1]], [point[2], point[2] + normalized_tan_vector[2]])

        # Plot bezier curve points and the tangent vectors at those points. MUST USE TRANSLATED DERIVATIVE VECTORS such that they start at their 
        # corresponding point
        # for point, tan_vector in zip(pos_bezier, trans_unit_der_bezier): 
        #     ax.scatter(point[0], point[1], point[2])
        #     ax.scatter(tan_vector[0], tan_vector[1], tan_vector[2])
        #     ax.plot([point[0], tan_vector[0]], [point[1], tan_vector[1]], [point[2], tan_vector[2]])


        # Plot bezier curve points and the tangent vectors at those points == WRONG -- didn't translate the derivative vectors such that 
        # they start at their corresponding point
        # for point, tan_vector in zip(pos_bezier, der_bezier): 
        #     normalized_tan_vector = F.normalize(tan_vector, p=2.0, dim=0)
        #     ax.scatter(point[0], point[1], point[2])
        #     ax.scatter(normalized_tan_vector[0], normalized_tan_vector[1], normalized_tan_vector[2])
        #     ax.plot([point[0], normalized_tan_vector[0]], [point[1], normalized_tan_vector[1]], [point[2], normalized_tan_vector[2]])

        # Plot bezier curve points and the normal vectors at those points
        # for point, norm_vector in zip(pos_bezier, normal_bezier): 
        #     ax.scatter(point[0], point[1], point[2])
        #     ax.scatter(norm_vector[0], norm_vector[1], norm_vector[2])
        #     ax.plot([point[0], norm_vector[0]], [point[1], norm_vector[1]], [point[2], norm_vector[2]])
        
        # for i, (der_vector, norm_vector) in enumerate(zip(trans_unit_der_bezier, normal_bezier)): 
        #     print("Should be 0: " + str(torch.cross(der_vector, norm_vector)))
        

        # Loop through each normal vec on Bezier curve. Rotate each normal by x degrees. Obtain points on circle
        # cylinder_pts = torch.zeros(self.num_samples, num_samples_on_circ, 3) 
        # for i, (point, norm_vector, tangent_vector) in enumerate(zip(pos_bezier, normal_bezier, trans_unit_der_bezier)): 
        #     for j in range(num_samples_on_circ): 
        #         # print("Rotation Angle: " + str(rotation_angle + (j - 1) * (rotation_angle)))
        #         cylinder_pts[ i:, j, :] = self.getRotatedVector(rotation_angle + (j - 1) * (rotation_angle), F.normalize(tangent_vector, p=2.0, dim=0), norm_vector)
        #         cylinder_pts[ i:, j, :] = F.normalize(cylinder_pts[ i:, j, :], p=2.0, dim=0)
        #         # Translate cylinder vectors such that they start at their corresponding point 
        #         cylinder_pts[i, j, 0] += point[0]
        #         cylinder_pts[i, j, 1] += point[1]
        #         cylinder_pts[i, j, 2] += point[2]
                

        # cylinder_pts *= radius
        # print("\n Points on cylinder surface: ")
        # print(cylinder_pts)

        # print("\n Example point on circle: \n")
        # test_circle_vector = cylinder_pts[0, 0, :]
        # print(test_circle_vector)
        # ax.scatter(test_circle_vector[0], test_circle_vector[1], test_circle_vector[2])


        # Somehow graph each tensor to visualize it
        # Loop through each circle. Loop through each point on circle. Plot those points
        # for i, (point, norm_vector) in enumerate(zip(pos_bezier, normal_bezier)): 
        #     ax.scatter(point[0], point[1], point[2])
        #     for j in range(num_samples_on_circ):
        #         # print("\n Current cylinder vector: " + str(cylinder_pts[i, j, :]))
        #         circle_vector = cylinder_pts[i, j, :]
        #         ax.scatter(circle_vector[0], circle_vector[1], circle_vector[2])
        #         # print("\n PLOTTED! ")
        #         ax.plot([point[0], circle_vector[0]], [point[1], circle_vector[1]], [point[2], circle_vector[2]])

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


a.getBezierCurveCylinder(test_control_points2, 0.05)
