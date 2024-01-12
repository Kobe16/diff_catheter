import random
import numpy as np
import cv2

import test_transforms
import test_bezier_interspace_transforms
from test_optimize_executor import ReconstructionOptimizeScriptExecutor

import sys
import os
# getting the name of the directory where the current file is present.
current_directory = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent_directory = os.path.dirname(current_directory)
# adding the parent directory to the sys.path.
sys.path.append(parent_directory)
# now we can import the module in the parent directory.
from bezier_set import BezierSet

class CCCatheter(): 
    def __init__(self, p_0, l, r, n_mid_points, n_iter, n_reconst_iters, verbose=1):
        """
        Args:
            p_0 ((3,) numpy array): start point of catheter
            l (float): length of catheter
            r (float): cross section radius of catheter
            n_mid_points (int): number of middle control points
            n_iter (int): number of total iteration of optimization
            n_reconst_iters: number of total iterations of reconstruction optimization
            verbose (0, 1, or 2): amount of verbosity

        Attributes:
            mode (int or None): Indicates the number of Degrees of Freedom (DoF).
            weight_matrix (numpy.ndarray): Matrix for weight calculations. Will take submatrix
                                           to use for calculations if necessary. 
            params (numpy.ndarray): Array to store catheter parameters over iterations.
            bezier_config ((1,6) numpy array): current bezier configuration (2nd and 3rd 
                                               control points respectively) of catheter

        """
        self.p_0 = p_0
        self.l = l
        self.r = r

        self.n_iter = n_iter
        self.n_reconst_iters = n_reconst_iters
        # self.verbose = verbose

        if n_mid_points == 0:
            self.n_mid_points = 0
        else:
            if n_mid_points % 2 == 0:
                self.n_mid_points = n_mid_points + 1
            else:
                self.n_mid_points = n_mid_points

        self.mode = None
        self.weight_matrix = np.zeros((3, 3))
        self.params = np.zeros((self.n_iter + 2, 5))
        # self.p3d_poses = np.zeros((self.n_iter + 2, self.n_mid_points + 1, 3))
        # self.p2d_poses = np.zeros((self.n_iter + 2, self.n_mid_points + 1, 2))
        self.bezier_config = np.zeros((6, 1))

    def set_camera_params(self, fx, fy, cx, cy, size_x, size_y, camera_extrinsics):
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
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.size_x = size_x
        self.size_y = size_y
        self.camera_extrinsics = camera_extrinsics


    def set_bezier_default_config(self): 
        """
        Set parameters for the default bezier configuration for the 2nd and 3rd control points. 
        Set the 3rd control point as distance l away from p_0, in the x-direction. 
        Set the 2nd control point as distance l/2 away from p_0, in the x-direction. 
        """
        self.bezier_default = np.zeros((6, 1))
        self.bezier_default[0, 0] = self.l / 2
        self.bezier_default[3, 0] = self.l

    def set_ux_uy_default_config(self, ux_default, uy_default): 
        """
        Set parameters for the default actuation state configuration
        """
        self.ux_default = ux_default
        self.uy_default = uy_default
        

    def set_1dof_params(self, phi, u):
        """
        Set parameters for 1 Degree of Freedom catheter

        Args:
            phi (radians as float): phi parameter
            u (float): tendon length (responsible for catheter bending)
        
        Note:
            mode = 1 indicates 1 DoF
        """
        self.mode = 1
        self.phi = phi
        self.u = u


    def set_2dof_params(self, ux, uy):
        """
        Set parameters for 2 Degrees of Freedom catheter

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
        
        Note:
            mode = 2 indicates 2 DoF
        """
        self.mode = 2
        self.ux = ux
        self.uy = uy

        self.params[0, 0] = self.ux
        self.params[0, 1] = self.uy

    def set_3dof_params(self, ux, uy, l):
        """
        Set parameters for 3 Degrees of Freedom catheter

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            l (float): length of bending portion of the catheter (responsible for insertion)
        
        Note:
            mode = 3 indicates 3 DoF
        """
        self.mode = 3
        self.ux = ux
        self.uy = uy
        self.l = l

        self.params[0, 0] = self.ux
        self.params[0, 1] = self.uy
        self.params[0, 2] = self.l

    def get_1dof_params(self):
        """
        Get parameters for 1 Degree of Freedom catheter

        Params:
            phi (radians as float): phi parameter
            u (float): tendon length (responsible for catheter bending)
        
        """
        return [self.phi, self.u]


    def get_2dof_params(self):
        """
        Get parameters for 2 Degrees of Freedom catheter

        Params:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
        
        """
        return self.params

    def get_3dof_params(self):
        """
        Get parameters for 3 Degrees of Freedom catheter

        Params:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            l (float): length of bending portion of the catheter (responsible for insertion)
        
        """

        return self.params
    

    def set_weight_matrix(self, w1=0, w2=0, w3=0):
        """
        Set weight matrix. The weight matrix is a 3x3 diagonal matrix. 
        The n-th diagonal term corresponds to the damping weight of the n-th DoF control feedback.

        Args:
            w1 (float): 1st DoF damping weight
            w2 (float): 2nd DoF damping weight
            w3 (float): 3rd DoF damping weight
        """
        self.weight_matrix = np.eye(3)
        self.weight_matrix[0, 0] = w1
        self.weight_matrix[1, 1] = w2
        self.weight_matrix[2, 2] = w3

    def transform_1_dof(self, s=1): 
        """
        1DoF constant curvature transform given the start point, the parameters, and the s value

        Args:
            s (float from 0 to 1 inclusive): s value representing position on the CC curve
            target (bool): whether to use target parameters for transform
       
        Note:
            mode = 1 indicates 1 DoF
        """
        if self.mode != 1:
            print('[ERROR] [CCCatheter] Mode incorrect. Current mode: ', str(self.mode))
            exit()

        # if target:
        #     if not self.u_target:
        #         print('[ERROR] [CCCatheter] 1 DOF target not set.')
        #         exit()

            # return transforms.cc_transform_1dof(self.p_0, self.phi, self.u_target, self.l, self.r, s)

        # else:
        return test_transforms.cc_transform_1dof(self.p_0, self.phi, self.u, self.l, self.r, s)

    def transform_2dof(self, s=1):
        """
        2DoF constant curvature transform given the start point, the parameters, and the s value

        Args:
            s (float from 0 to 1 inclusive): s value representing position on the CC curve
            target (bool): whether to use target parameters for transform
       
        Note:
            mode = 2 indicates 2 DoF
        """
        if self.mode != 2:
            print('[ERROR] [CCCatheter] Mode incorrect. Current mode: ', str(self.mode))
            exit()

        # if target:
        #     if not (self.ux_target and self.uy_target):
        #         print('[ERROR] [CCCatheter] 2 DOF target not set.')
        #         exit()

        #     return transforms.cc_transform_3dof(self.p_0, self.ux_target, self.uy_target, self.l, self.r, s)

        # else:
        return test_transforms.cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)

    def transform_3dof(self, s=1):
        """
        3DoF constant curvature transform given the start point, the parameters, and the s value

        Args:
            s (float from 0 to 1 inclusive): s value representing position on the CC curve
            target (bool): whether to use target parameters for transform
       
        Note:
            mode = 3 indicates 3 DoF
        """
        if self.mode != 3:
            print('[ERROR] [CCCatheter] Mode incorrect. Current mode: ', str(self.mode))
            exit()

        # if target:
        #     if not (self.ux_target and self.uy_target and self.l_target):
        #         print('[ERROR] [CCCatheter] 3 DOF target not set.')
        #         exit()

        #     return transforms.cc_transform_3dof(self.p_0, self.ux_target, self.uy_target, self.l_target, self.r, s)

        # else:
        return test_transforms.cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)



    def calculate_cc_points(self, current_iter=0, init=False, target=False):
        """
        Calculate the list of points on the constant curvature curve given n_mid_points
            n_mid_points is the number of points between the two end points of the CC curve

        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            init (bool): whether this is the first iteration
                This is only used for parameter recording
            target (bool): whether to use target parameters for transform
        """
        if self.n_mid_points == 0:
            self.s_list = [1]
        else:
            self.s_list = np.linspace(0, 1, self.n_mid_points + 2)[1:]

        # if target:
        #     self.target_cc_pt_list = []

        #     for i, s, in enumerate(self.s_list):

        #         if self.mode == 1:
        #             p = self.transform_1dof(s, target=True)
        #         elif self.mode == 2:
        #             p = self.transform_2dof(s, target=True)
        #         elif self.mode == 3:
        #             p = self.transform_3dof(s, target=True)
        #         else:
        #             print('[ERROR] [CCCatheter] Mode invalid')
        #             exit()

        #         self.target_cc_pt_list.append(p)

        #         self.p3d_poses[-1, i, :] = p

        #         if self.verbose > 0:
        #             print('Target CC Point ' + str(i + 1) + ': ')
        #             print('    s = ', s)
        #             print('    p = ', p)

        # else:
        self.cc_pt_list = []

        for i, s in enumerate(self.s_list):

            if self.mode == 1:
                p = self.transform_1dof(s)
            elif self.mode == 2:
                p = self.transform_2dof(s)
            elif self.mode == 3:
                p = self.transform_3dof(s)
            else:
                print('[ERROR] [CCCatheter] Mode invalid')
                exit()

            self.cc_pt_list.append(p)

            # if self.verbose > 0:
            #     print('CC Point ' + str(i + 1) + ': ')
            #     print('    s = ', s)
            #     print('    p = ', p)

            if current_iter < 0:
                continue

            # if init:
            #     self.p3d_poses[0, i, :] = p
            # else:
            #     self.p3d_poses[current_iter + 1, i, :] = p

    def calculate_beziers_control_points(self):
        """
        Given the list of points on the constant curvature curve, calculate the control points for a
            number of Bezier curves. The number of Bezier curves is determined by number of cc points // 2 

        Note: 
            Must run calculate_cc_points before this to get points along the curve. 
        """
        if not self.cc_pt_list:
            print('[ERROR] [CCCatheter] self.cc_pt_list invalid. Run calculate_cc_points() first')
            exit()

        n_beziers = int(len(self.cc_pt_list) / 2)
        self.bezier_set = BezierSet(n_beziers)

        cc_pt_list_with_p_0 = [self.p_0] + self.cc_pt_list

        for i in range(n_beziers):
            p_start = cc_pt_list_with_p_0[2 * i]
            p_mid = cc_pt_list_with_p_0[2 * i + 1]
            p_end = cc_pt_list_with_p_0[2 * (i + 1)]

            c = (p_mid - (p_start / 4) - (p_end / 4)) * 2
            c1 = 4 / 3 * p_mid - 1 / 3 * p_end
            c2 = 4 / 3 * p_mid - 1 / 3 * p_start

            self.bezier_set.enter_spec(p_start, p_end, c1, c2)

            # if self.verbose > 1:
            #     print('Bezier ' + str(i) + ': ')
            #     print('    p_start = ', p_start)
            #     print('    p_mid   = ', p_mid)
            #     print('    p_end   = ', p_end)

    def render_beziers(self,
                       curve_specs_path,
                       img_save_path,
                       target_specs_path=None,
                       viewpoint_mode=1,
                       transparent_mode=0):
        """
        Render Bezier curves according to the curve specs.

        Args:
            curve_specs_path (path string to npy file): curve specs is a (n, 3) numpy array where each
                row specifies the start point, middle control point, and end point of a Bezier curve
            img_save_path (path string to png file): path to save rendered image
            target_specs_path (path string to npy file): target specs is a (n, 3) numpy array where each
                row specifies the 3D position of a target point. If this path is set to None,
                target points will be not rendered
            viewpoint_mode (1 or 2): camera view of rendered image, 1 for endoscopic view, 2 for side view
            transparent_mode (0 or 1): whether to make the background transparent for the rendered image,
                0 for not transparent, 1 for transparent
        """
        if not self.bezier_set:
            print('[ERROR] [CCCatheter] self.bezier_set invalid. Run calculate_beziers_control_points() first')
            exit()

        self.bezier_set.print_specs()
        self.bezier_set.write_specs(curve_specs_path)
        self.bezier_set.render(img_save_path, target_specs_path, viewpoint_mode, transparent_mode)



    def calculate_p_diffs(self): 
        """
        Calculate difference between current bezier configuration and default bezier configuration. 
        Basically, find difference for the 2nd and 3rd Bezier control ponints. 
        """

        self.p_diffs = np.zeros((6, 1))
        self.p_diffs = self.bezier_default - self.bezier_config

    
    def get_bezier_reconstruction(self, img_get_path, img_save_path): 
        """
        Get the bezier params of a catheter from a picture of the catheter. 
        Call's Kobe's summer 2023 catheter reconstruction script
        """
        para_init = self.bezier_config.reshape((6,))

        optimize_executor = ReconstructionOptimizeScriptExecutor(
            self.p_0,
            para_init,
            self.n_reconst_iters,
            img_get_path,
            img_save_path
        )

        optimize_executor.execute()

    def get_2dof_bezier_interspace_ux_uy(self, bezier_reconst, bezier_t0, ux_t0, damping_const): 
        """
        Inverse kinematics function mapping from Bezier configuration to actuation state.

        Args:
            bezier_reconst (numpy array): Reconstructed Bezier curve.
            bezier_t0 (numpy array): Initial Bezier curve.
            ux_t0 (numpy array): Initial actuation state.
            damping_const (float): Damping constant.

        Returns:
            ux (numpy array): Actuation state for inputted Bezier configuration.
        """

        self.calculate_p_diffs()
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        # TODO: replace this inverse kinematics math with the damped least squares method
        # TODO: change name of this method. It shouldn't be called update.... because its
        #       its goal is not to update parameters to simulate movement. Its goal is to 
        #       just convert between the difference interspaces. I think I originally named
        #       it 'update' because I just copied the code over from cc_catheter.py

        # Calculate jacobian for current state of catheter
        J = test_bezier_interspace_transforms.calculate_jacobian_2dof_ux_uy(self.p_0, self.ux, self.uy, self.l, self.r)

        # Get Jacobian inverse J_inv
        # J_inv = np.linalg.inv(J)

        # u_delta = damping_const * J_inv * (bezier_t0 - bezier_reconst)

        # ux = ux_t0 - u_delta

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix[:2, :2]
        print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_ux = d[0, 0]
        d_uy = d[1, 0]

        ux_old = self.ux
        uy_old = self.uy

        # self.ux += d_ux
        # self.uy += d_uy

        self.ux = self.ux_default - d_ux
        self.uy = self.uy_default - d_uy

        return self.ux, self.uy
    
    def get_2dof_ux_uy_interspace_bezier(self): 
        """
        Forward kinematics function mapping actuation state to Bezier configuration
        """

        bezier_config = np.zeros((6, 1))

        p_2 = test_bezier_interspace_transforms.cc_transform_3dof(
            self.p_0,
            self.ux,
            self.uy,
            self.l,
            self.r,
            1
        )
        curve_midpoint = test_bezier_interspace_transforms.cc_transform_3dof(
            self.p_0,
            self.ux,
            self.uy,
            self.l,
            self.r,
            0.5
        )

        p_1 = 2 * curve_midpoint - self.p_0 / 2  - p_2 / 2
        
        bezier_config[0:3, 0] = p_1
        bezier_config[3:6, 0]  = p_2

        return bezier_config
    
    def get_2dof_ux_uy_num_steps_ago(self, ux_reconst, uy_reconst, current_iter, num_steps): 
        """
        Use history of controls to get the reconstructed actuation state from num steps ago

        Args: 
            ux_reconst (float): actuation state ux that created from bezier reconstruction
            uy_reconst (float): actuation state uy that created from bezier reconstruction
            current_iter (int): current frame that we are reconstructed and performing loss function on
            num_steps (int): number of steps to travel backwards in frames in order to get 
                             sum of actuation changes since that time. 

        """

        ux_control_changes_sum = sum(self.ux[current_iter - num_steps : current_iter])
        uy_control_changes_sum = sum(self.ux[current_iter - num_steps : current_iter])

        ux_num_steps_ago = ux_reconst - ux_control_changes_sum
        uy_num_steps_ago = uy_reconst - uy_control_changes_sum

        return ux_num_steps_ago, uy_num_steps_ago


     


