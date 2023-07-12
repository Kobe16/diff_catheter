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

    def getFrenetFrame(der_vector, double_der_vector): 
        normalized_a = F.normalize(der_vector, p=2, dim=2)
        normalized_b = F.normalize(normalized_a + double_der_vector, p=2)
        rotational_axis_vector = F.normalize(torch.cross(normalized_a, normalized_b))
        normal_vector = F.normalize(torch.cross(rotational_axis_vector, normalized_a))

        return normal_vector

    def getBezierCurveCylinder(self, control_pts, radius): 
        
        self.num_samples = 9
        #self.num_samples = 200
        P0 = control_pts[0, :]
        P1 = control_pts[1, :]
        P2 = control_pts[2, :]
        P3 = control_pts[3, :]

        print(P0)
        print(P1)
        print(P2)
        print(P3)

        sample_list = torch.linspace(0, 1, self.num_samples)

        print("\n Sample list: ")
        print(sample_list)

        # Get positions and normals [NOTE: SHOULD be tangents?] from samples along bezier curve
        pos_bezier = torch.zeros(self.num_samples, 3)
        der_bezier = torch.zeros(self.num_samples, 3)
        double_der_bezier = torch.zeros(self.num_samples, 3)
        for i, s in enumerate(sample_list):
            pos_bezier[i, :] = (1 - s)**3 * P0 + 3 * s * (1 - s)**2 * \
                P1 + 3 * (1 - s) * s**2 * P2 + s**3 * P3
            der_bezier[i, :] = -(1 - s)**2 * P0 + ((1 - s)**2 - 2 * s * (1 - s)) * P1 + (-s**2 + 2 *
                                                                                    (1 - s) * s) * P2 + s**2 * P3
            double_der_bezier[i, :] = 6 * (1 - s) * (P2 - 2*P1 + P0) + 6 * (P3 - 2*P2 + P1) * s
            
        # print("\n Positions of bezier points: ")
        # print(pos_bezier)
        # print("\n Derivatives of bezier points: ")
        # print(der_bezier)
        # print("\n Double Derivatives of bezier points: ")
        # print(double_der_bezier)

        # plt.xlim(0, 5)
        # plt.ylim(0, 5)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for point in pos_bezier: 
            # print(point)
            ax.scatter(point[0], point[1], point[2])
        
        plt.show()

        # Calculate 3d normals at each point (RMF method)
        # bezier_normal_vectors 


        # Rotate each 3d normal by x degrees. Obtain points on circle

        # Somehow graph each tensor to visualize it

        return 0


a = reconstructCurve()
test_control_points = torch.tensor([[0., 0., 0.], 
                                    [0., 1., 2.5], 
                                    [0., 2., 0.5], 
                                    [0., 3., 3.]])

a.getBezierCurveCylinder(test_control_points, 0.05)