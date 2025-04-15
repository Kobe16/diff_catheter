import random
import numpy as np
import cv2

import transforms
import bezier_interspace_transforms
from bezier_set import BezierSet


class CCCatheter:
    def __init__(self, p_0, l, r, loss_2d, tip_loss, n_mid_points, n_iter, verbose=1):
        """
        Args:
            p_0 ((3,) numpy array): start point of catheter
            l (float): length of catheter
            r (float): cross section radius of catheter
            loss_2d (bool): whether to use 2D loss
            tip_loss (bool): whether to use tip loss
            n_mid_points (int): number of middle control points
            n_iter (int): number of total iteration of optimization
            verbose (0, 1, or 2): amount of verbosity
        """
        self.p_0 = p_0
        self.l = l
        self.r = r

        self.loss_2d = loss_2d
        self.tip_loss = tip_loss
        self.n_iter = n_iter
        self.verbose = verbose

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
        self.p3d_poses = np.zeros((self.n_iter + 2, self.n_mid_points + 1, 3))
        self.p2d_poses = np.zeros((self.n_iter + 2, self.n_mid_points + 1, 2))
        
        self.noise_list = []

    def set_1dof_params(self, phi, u):
        """
        Set parameters

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
        Set parameters

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
        
        Note:
            mode = 2 indicates 2 DoF
        """
        self.mode = 2
        self.ux = ux
        self.uy = uy
        
        self.ux_theory = ux
        self.uy_theory = uy

        self.params[0, 0] = self.ux
        self.params[0, 1] = self.uy

    def set_3dof_params(self, ux, uy, l):
        """
        Set parameters

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
        
        self.ux_theory = ux
        self.uy_theory = uy

        self.params[0, 0] = self.ux
        self.params[0, 1] = self.uy
        self.params[0, 2] = self.l

    def set_1dof_targets(self, u_target):
        """
        Set target

        Args:
            u_target (float): tendon length (responsible for catheter bending)
        """
        self.u_target = u_target

    def set_2dof_targets(self, ux_target, uy_target):
        """
        Set targets

        Args:
            ux_target (float): 1st pair of tendon length (responsible for catheter bending)
            uy_target (float): 2nd pair of tendon length (responsible for catheter bending)
        """
        self.ux_target = ux_target
        self.uy_target = uy_target

        self.params[-1, 0] = self.ux_target
        self.params[-1, 1] = self.uy_target

    def set_3dof_targets(self, ux_target, uy_target, l_target):
        """
        Set targets

        Args:
            ux_target (float): 1st pair of tendon length (responsible for catheter bending)
            uy_target (float): 2nd pair of tendon length (responsible for catheter bending)
            l_target (float): length of bending portion of the catheter (responsible for insertion)
        """
        self.ux_target = ux_target
        self.uy_target = uy_target
        self.l_target = l_target

        self.params[-1, 0] = self.ux_target
        self.params[-1, 1] = self.uy_target
        self.params[-1, 2] = self.l_target

    def set_2d_targets(self, x_targets, y_targets):
        """
        Set targets for 2D loss

        Args:
            x_target (int): horizontal target pixel location of end effector
            y_target (int): vertical target pixel location of end effector
        
        Notes:
            1) Only works when loss_2d is set to True during instantiation
            2) The arguments can be float but would be rounded to int anyway.
        """
        if not self.loss_2d:
            print('[ERROR] [CCCatheter] Not initialized with 2D Loss')
            exit()

        self.target_cc_pt_list_2d = []

        for i, (x, y) in enumerate(zip(x_targets, y_targets)):
            p_2d_target = np.array([round(x), round(y)])
            self.target_cc_pt_list_2d.append(p_2d_target)
            self.p2d_poses[-1, i, :] = p_2d_target

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

    def transform_1dof(self, s=1, target=False):
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

        if target:
            if not self.u_target:
                print('[ERROR] [CCCatheter] 1 DOF target not set.')
                exit()

            return transforms.cc_transform_1dof(self.p_0, self.phi, self.u_target, self.l, self.r, s)

        else:
            return transforms.cc_transform_1dof(self.p_0, self.phi, self.u, self.l, self.r, s)

    def transform_2dof(self, s=1, target=False, theory=False):
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

        if target:
            if not (self.ux_target and self.uy_target):
                print('[ERROR] [CCCatheter] 2 DOF target not set.')
                exit()

            return transforms.cc_transform_3dof(self.p_0, self.ux_target, self.uy_target, self.l, self.r, s)

        else:
            if theory:
                return transforms.cc_transform_3dof(self.p_0, self.ux_theory, self.uy_theory, self.l, self.r, s)
            return transforms.cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)

    def transform_3dof(self, s=1, target=False, theory=False):
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

        if target:
            if not (self.ux_target and self.uy_target and self.l_target):
                print('[ERROR] [CCCatheter] 3 DOF target not set.')
                exit()

            return transforms.cc_transform_3dof(self.p_0, self.ux_target, self.uy_target, self.l_target, self.r, s)

        else:
            if theory:
                return transforms.cc_transform_3dof(self.p_0, self.ux_theory, self.uy_theory, self.l, self.r, s)
            return transforms.cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)
        
    def u_to_cc(self, theory=False):
        cc_pt_list = []
        for i, s in enumerate(self.s_list):
                if self.mode == 1:
                    p = self.transform_1dof(s)
                elif self.mode == 2:
                    p = self.transform_2dof(s, theory=theory)
                elif self.mode == 3:
                    p = self.transform_3dof(s, theory=theory)
                else:
                    print('[ERROR] [CCCatheter] Mode invalid')
                    exit()
                cc_pt_list.append(p)
        return cc_pt_list
        

    def calculate_cc_points(self, current_iter=0, init=False, target=False, noise_percentage=0):
        """
        Calculate the list of points on the constant curvature curve given n_mid_points
            n_mid_points is the number of points between the two end points of the CC curve

        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            init (bool): whether this is the first iteration
                This is only used for parameter recording
            target (bool): whether to calculate target constant curvature points
            
        Results:
            self.cc_pt_list (list of (3,) np array, [c_mid, c_end]): 
                list of 3D points on the constant curvature curve (ground truth of simulated catheter)
            self.cc_pt_list_theory: theoretical, without consideration of noise (user-capable data)
            self.target_cc_pt_list: target
        """
        if self.n_mid_points == 0:
            self.s_list = [1]
        else:
            self.s_list = np.linspace(0, 1, self.n_mid_points + 2)[1:]

        # ------ Calculate the target constant curvature points ------
        if target:
            self.target_cc_pt_list = [] # Everytime calculate_cc_points() is called, target_cc_pt_list is reset

            for i, s, in enumerate(self.s_list):

                if self.mode == 1:
                    p = self.transform_1dof(s, target=True)
                elif self.mode == 2:
                    p = self.transform_2dof(s, target=True)
                elif self.mode == 3:
                    p = self.transform_3dof(s, target=True)
                else:
                    print('[ERROR] [CCCatheter] Mode invalid')
                    exit()

                self.target_cc_pt_list.append(p)

                self.p3d_poses[-1, i, :] = p

                if self.verbose > 0:
                    print('Target CC Point ' + str(i + 1) + ': ')
                    print('    s = ', s)
                    print('    p = ', p)

        else:
            # constant curvature points in theory (for initial guess in reconstruction)
            self.cc_pt_list_theory = []
            # constant curvature points in reality (simulation) ------   
            self.cc_pt_list = []
            
            # ------ Calculate the constant curvature points without noise ------
            # cc_pt_list_no_noise = []
            # for i, s in enumerate(self.s_list):
            #     if self.mode == 1:
            #         p = self.transform_1dof(s)
            #     elif self.mode == 2:
            #         p = self.transform_2dof(s)
            #     elif self.mode == 3:
            #         p = self.transform_3dof(s)
            #     else:
            #         print('[ERROR] [CCCatheter] Mode invalid')
            #         exit()
            #     cc_pt_list_no_noise.append(p)
                   
            self.cc_pt_list_theory = self.u_to_cc(theory=True)
            
            # ------ Calculate the constant curvature points with noise ------
            # Add noise to the initial control parameters (ux, uy, l)
            if init:
                self.ux = random.gauss(self.ux, noise_percentage * self.ux)
                self.uy = random.gauss(self.uy, noise_percentage * self.uy)
                # self.l = random.gauss(self.l, noise_percentage * self.l)
            
                # cc_pt_list_noisy = []

                # for i, s in enumerate(self.s_list):

                #     if self.mode == 1:
                #         p = self.transform_1dof(s)
                #     elif self.mode == 2:
                #         p = self.transform_2dof(s)
                #     elif self.mode == 3:
                #         p = self.transform_3dof(s)
                #     else:
                #         print('[ERROR] [CCCatheter] Mode invalid')
                #         exit()

                #     cc_pt_list_noisy.append(p)   
            self.cc_pt_list = self.u_to_cc()

            # if self.verbose > 0:
            #     print('CC Point ' + str(i + 1) + ': ')
            #     print('    s = ', s)
            #     print('    p = ', p)

            # if current_iter < 0:
            #     continue

            # if init:
            #     self.p3d_poses[0, i, :] = p
            # else:
            #     self.p3d_poses[current_iter + 1, i, :] = p

    def convert_bezier_to_cc(self, optimized_bezier_specs, current_iter=0):
        """
        Convert a Bezier curve back to a constant curvature curve

        Args:
            optimized_bezier_specs ((2, 3) numpy array): the 1st row specifies the middle control
                point of the Bezier curve; the 2nd row specifies the end points of the Bezier curve
            current_iter (int): current iteration in optimization.
                    This is only used for parameter recording

        Note:
            This function is used to read results from Reconstruction and only supports converting
                1 Bezier curve to a CC curve with 1 mid point 
        """
        if len(self.cc_pt_list) != 2:
            print('[ERROR] Reconstruction is not compatible with more than 1 mid points')
            exit()

        p_start = self.p_0
        c = optimized_bezier_specs[0, :]
        p_end = optimized_bezier_specs[1, :]
        p_mid = (c / 2) + (p_start / 4) + (p_end / 4)

        self.cc_pt_list[0] = p_mid
        self.cc_pt_list[1] = p_end

        self.p3d_poses[current_iter + 1, 0, :] = p_mid
        self.p3d_poses[current_iter + 1, 1, :] = p_end   

    def convert_cc_points_to_2d(self, current_iter=0, init=False, target=False):
        """
        Convert points on the constant curvature curve to 2D points on the image taken by camera given the camera info

        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            init (bool): whether this is the first iteration
                This is only used for parameter recording
            target (bool): whether to calculate target constant curvature points in 2D
            
        Results:
            in_view (bool): whether the target points are in view of the camera
            self.cc_pt_list_2d
            self.target_cc_pt_list_2d 
        """
        #if not self.loss_2d:
        #    print('[ERROR] [CCCatheter] Not initialized with 2D Loss')
        #    exit()

        in_view = True

        if target:
            if not self.target_cc_pt_list:
                print(
                    '[ERROR] [CCCatheter] self.target_cc_pt_list invalid. Run calculate_cc_points() with target=True first'
                )
                exit()

            self.target_cc_pt_list_2d = []

            for i, p in enumerate(self.target_cc_pt_list):

                p_2d = transforms.world_to_image_transform(p, self.camera_extrinsics, self.fx, self.fy, self.cx,
                                                           self.cy)
                p_2d[0] = round(self.size_x - p_2d[0])
                p_2d[1] = round(p_2d[1])

                if p_2d[0] >= self.size_x or p_2d[0] < 0 or p_2d[1] >= self.size_y or p_2d[1] < 0:
                    print('[ERROR] [CCCatheter] Target falls outside of image. Target 2D position = ', p_2d)
                    exit()

                self.target_cc_pt_list_2d.append(p_2d)

                self.p2d_poses[-1, i, :] = p_2d

                if self.verbose > 1:
                    print('2D Target CC Point ' + str(i + 1) + ': ')
                    print('    p = ', p_2d)

        else:
            if not self.cc_pt_list:
                print('[ERROR] [CCCatheter] self.cc_pt_list invalid. Run calculate_cc_points() first')
                exit()

            self.cc_pt_list_2d = []

            for i, p in enumerate(self.cc_pt_list):
                p_2d = transforms.world_to_image_transform(p, self.camera_extrinsics, self.fx, self.fy, self.cx,
                                                           self.cy)
                p_2d[0] = self.size_x - p_2d[0]

                if p_2d[0] >= self.size_x or p_2d[0] < 0 or p_2d[1] >= self.size_y or p_2d[1] < 0:
                    in_view = False

                self.cc_pt_list_2d.append(p_2d)

                if self.verbose > 1:
                    print('2D CC Point ' + str(i + 1) + ': ')
                    print('    p = ', p_2d)

                if current_iter < 0:
                    continue

                if init:
                    self.p2d_poses[0, i, :] = p_2d
                else:
                    self.p2d_poses[current_iter + 1, i, :] = p_2d

        return in_view
    
    def convert_bezier_points_to_2d(self, current_iter=0, target=False, use_reconstruction=False):
        """
        Convert points on the constant curvature curve to 2D points on the image taken by camera given the camera info

        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            init (bool): whether this is the first iteration
                This is only used for parameter recording
            target (bool): whether to calculate target constant curvature points in 2D
            
        Results:
            in_view (bool): whether the target points are in view of the camera
            self.bezier_params_list_2d
            self.target_cc_pt_list_2d 
        """
        in_view = True

        # ------ Target Bezier points ------
        if target:
            if not self.target_bezier_params_list:
                print(
                    '[ERROR] [CCCatheter] self.target_bezier_params_list invalid. Run calculate_beziers_control_points(target=True) first'
                )
                exit()

            self.target_bezier_params_list_2d = []

            for i, p in enumerate(self.target_bezier_params_list):

                p_2d = transforms.world_to_image_transform(p, self.camera_extrinsics, self.fx, self.fy, self.cx,
                                                           self.cy)
                p_2d[0] = round(self.size_x - p_2d[0])
                p_2d[1] = round(p_2d[1])

                if p_2d[0] >= self.size_x or p_2d[0] < 0 or p_2d[1] >= self.size_y or p_2d[1] < 0:
                    print('[ERROR] [CCCatheter] Target falls outside of image. Target 2D position = ', p_2d)
                    exit()

                self.target_bezier_params_list_2d.append(p_2d)

                self.p2d_poses[-1, i, :] = p_2d

                if self.verbose > 1:
                    print('2D Target Bezier Point ' + str(i + 1) + ': ')
                    print('    p = ', p_2d)
        # ------ Optimized Bezier points ------ 
        elif use_reconstruction:
            if not hasattr(self, 'bezier_params_optimized'):
                print('[ERROR] [CCCatheter] self.bezier_params_optimized invalid. Run reconstruction process first')
                exit()
            
            self.bezier_params_optimized_2d = []

            for i, p in enumerate(self.bezier_params_optimized):
                p_2d = transforms.world_to_image_transform(p, self.camera_extrinsics, self.fx, self.fy, self.cx,
                                                            self.cy)
                p_2d[0] = self.size_x - p_2d[0]

                if p_2d[0] >= self.size_x or p_2d[0] < 0 or p_2d[1] >= self.size_y or p_2d[1] < 0:
                    in_view = False

                self.bezier_params_optimized_2d.append(p_2d)

        # ------ Bezier points ------
        else:
            if self.bezier_params_list:
                self.bezier_params_list_2d = []

                for i, p in enumerate(self.bezier_params_list):
                    p_2d = transforms.world_to_image_transform(p, self.camera_extrinsics, self.fx, self.fy, self.cx,
                                                                self.cy)
                    p_2d[0] = self.size_x - p_2d[0]

                    if p_2d[0] >= self.size_x or p_2d[0] < 0 or p_2d[1] >= self.size_y or p_2d[1] < 0:
                        in_view = False

                    self.bezier_params_list_2d.append(p_2d)

                    if self.verbose > 1:
                        print('2D Bezier Point ' + str(i + 1) + ': ')
                        print('    p = ', p_2d)

                    if current_iter < 0:
                        continue

                    # if init:
                    #     self.p2d_poses[0, i, :] = p_2d
                    # else:
                    #     self.p2d_poses[current_iter + 1, i, :] = p_2d

        return in_view

    def calculate_bezier_specs(self, init=False):
        """
        Calculate the Bezier specs for the current list of points on the constant curvature curve

        Returns:
            bezier_specs ((2, 3) numpy array): the 1st row is the middle control point of Bezier curve;
                the 2nd row is the end point of Bezier curve

        Note:
            This only works for n_mid_points = 1
        """
        # For the first iteration, use the theoretical bezier control points as initial guess
        if init:
            if len(self.cc_pt_list_theory) != 2:
                print('[ERROR] Reconstruction is not compatible with more than 1 mid points')
                exit()

            bezier_specs = np.zeros((2, 3))

            p_mid = self.cc_pt_list_theory[0]
            p_end = self.cc_pt_list_theory[1]

            c = (p_mid - (self.p_0 / 4) - (p_end / 4)) * 2

            bezier_specs[0, :] = c 
            bezier_specs[1, :] = p_end

            return bezier_specs

        # For the rest of the iterations, general case
        # if len(self.cc_pt_list) != 2:
        #     print('[ERROR] Reconstruction is not compatible with more than 1 mid points')
        #     exit()

        bezier_specs = np.zeros((2, 3))

        # p_mid = self.cc_pt_list[0]
        # p_end = self.cc_pt_list[1]
        
        p_mid = self.p_mid
        p_end = self.p_end

        c = (p_mid - (self.p_0 / 4) - (p_end / 4)) * 2

        bezier_specs[0, :] = c
        bezier_specs[1, :] = p_end

        return bezier_specs
    
    def write_bezier_specs(self, bezier_specs, use_reconstruction=False):
        """
        Results:
            self.bezier_params_optimized: result of reconstruction, user-capable bezier parameters.
            self.bezier_params_list: ground truth.
        """
        
        if use_reconstruction:
            self.bezier_params_optimized = []
            self.bezier_params_optimized.append(bezier_specs[0, :])
            self.bezier_params_optimized.append(bezier_specs[1, :])
        else:
            if not self.bezier_params_list:
                    print('[ERROR] [CCCatheter] self.bezier_params_list invalid. Run calculate_bezier_specs() first')
                    exit()
                    
            self.bezier_params_list[0] = bezier_specs[0, :]
            self.bezier_params_list[1] = bezier_specs[1, :]

    def write_target_specs(self, target_specs_path, show_mid_points=True):
        """
        Write target specs to be used for rendering with Blender

        Args:
            target_specs_path (path string to npy file): target specs is a (n, 3) numpy array where each
                row specifies the 3D position of a target point
            show_mid_points (bool): whether to show the midpoints of the catheter or only the end point
        """
        if not self.target_cc_pt_list:
            print(
                '[ERROR] [CCCatheter] self.target_cc_pt_list invalid. Run calculate_cc_points() with target=True first')
            exit()

        if show_mid_points:
            target_specs = np.zeros((len(self.target_cc_pt_list) + 1, 3))

            for i, p in enumerate(self.target_cc_pt_list):
                target_specs[i + 1, :] = self.target_cc_pt_list[i]

        else:
            target_specs = np.zeros((2, 3))
            target_specs[1, :] = self.target_cc_pt_list[-1]

        target_specs[0, :] = self.p_0

        np.save(target_specs_path, target_specs)

    def write_target_gt_specs(self, target_gt_specs_path):
        """
        Write target ground truth specs

        Args:
            target_gt_specs_path (path string to npy file): target_gt_specs is a (3, 3) numpy array;
                the 1st row is the start point;
                the 2nd row is the target middle control point;
                the 3rd row is the target end point 
        """
        target_gt_specs = np.zeros((3, 3))

        p_start = self.p_0
        p_end = self.target_cc_pt_list[-1]
        s_mid = 0.5

        if self.mode == 1:
            p_mid = self.transform_1dof(s_mid, target=True)
        elif self.mode == 2:
            p_mid = self.transform_2dof(s_mid, target=True)
        elif self.mode == 3:
            p_mid = self.transform_3dof(s_mid, target=True)
        else:
            print('[ERROR] [CCCatheter] Mode invalid')
            exit()

        c = (p_mid - (p_start / 4) - (p_end / 4)) * 2

        target_gt_specs[0, :] = p_start
        target_gt_specs[1, :] = c
        target_gt_specs[2, :] = p_end

        np.save(target_gt_specs_path, target_gt_specs)

    def write_reports(self, params_report_path, p3d_report_path, p2d_report_path):
        """
        Save the recorded parameters, 3D positions of control points, and 2D positions of control points

        Args:
            params_report_path (path string to npy file): self.params is a (n_iter + 2, 5) numpy array
                the first row records the initial parameters;
                the last row records the target parameters;
                and the intermediate rows record the parameters throughout the iterations.
                The 5 columns records the ux, uy, l, theta, phi parameters.
                If some parameters are not applicable for current method, they are left as 0
            p3d_report_path (path string to npy file)
            p2d_report_path (path string to npy file)
        """
        np.save(params_report_path, self.params)
        np.save(p3d_report_path, self.p3d_poses)
        np.save(p2d_report_path, self.p2d_poses)

    def get_params(self):
        """
        Returns:
            self.params ((n_iter + 2, 5) numpy array): the first row records the initial parameters;
                the last row records the target parameters;
                and the intermediate rows record the parameters throughout the iterations.
                The 5 columns records the ux, uy, l, theta, phi parameters.
                If some parameters are not applicable for current method, they are left as 0
        """
        return self.params

    def calculate_p_diffs(self, bezier=False):
        """
        Calculate the difference between current positions and target positions
        """
        # if use constant curvature points for feedback
        if not bezier: 
            if self.loss_2d:
                if not (self.cc_pt_list_2d and self.target_cc_pt_list_2d):
                    print(
                        '[ERROR] [CCCatheter] self.cc_pt_list_2d or self.target_cc_pt_list_2d invalid. Run calculate_cc_points() then convert_cc_points_to_2d() first'
                    )
                    exit()

                if self.tip_loss:
                    self.p_diffs = np.zeros((2, 1))
                    self.p_diffs[:, 0] = self.target_cc_pt_list_2d[-1] - self.cc_pt_list_2d[-1]

                else:
                    self.p_diffs = np.zeros((2 * len(self.cc_pt_list), 1))

                    for i, (p, p_target) in enumerate(zip(self.cc_pt_list_2d, self.target_cc_pt_list_2d)):
                        self.p_diffs[i * 2:(i + 1) * 2, 0] = (p_target - p)

            else:
                if not (self.cc_pt_list and self.target_cc_pt_list):
                    print(
                        '[ERROR] [CCCatheter] self.cc_pt_list or self.target_cc_pt_list invalid. Run calculate_cc_points() first'
                    )
                    exit()

                if self.tip_loss:
                    self.p_diffs = np.zeros((3, 1))
                    self.p_diffs[:, 0] = self.target_cc_pt_list[-1] - self.cc_pt_list[-1]

                else:
                    self.p_diffs = np.zeros((3 * len(self.cc_pt_list), 1))

                    for i, (p, p_target) in enumerate(zip(self.cc_pt_list, self.target_cc_pt_list)):
                        self.p_diffs[i * 3:(i + 1) * 3, 0] = (p_target - p)
        
        # if use bezier control points for feedback             
        else: 
            if self.loss_2d:
                # if not hasattr(self, 'bezier_params_list_2d'):
                #     print('[ERROR] [CCCatheter] 2D bezier parameters invalid.')
                #     exit()
                    
                # if use reconstruction result as feedback
                if hasattr(self, 'bezier_params_optimized_2d'):
                    bezier_params_list_2d = self.bezier_params_optimized_2d
                    print("Use reconstruction result as feedback")
                # if use ground truth as feedback
                else:
                    bezier_params_list_2d = self.bezier_params_list_2d
                    print("Use ground truth as feedback")
                    
                if self.tip_loss:
                    self.p_diffs = np.zeros((2, 1))
                    self.p_diffs[:, 0] = self.target_bezier_params_list_2d[-1] - bezier_params_list_2d[-1]
                    print("Calculating 2D bezier tip loss")

                else:
                    self.p_diffs = np.zeros((2 * len(bezier_params_list_2d), 1))

                    for i, (p, p_target) in enumerate(zip(bezier_params_list_2d, self.target_bezier_params_list_2d)):
                        self.p_diffs[i * 2:(i + 1) * 2, 0] = (p_target - p)

            # --- 3D loss ---
            else:
                # if not self.target_bezier_params_list or not self.bezier_params_optimized:
                #     print(
                #         '[ERROR] [CCCatheter] self.target_bezier_params_list invalid. Run calculate_beziers_control_points(target=True) first'
                #     )
                #     exit()
                    
                # if not self.bezier_params_list or not self.bezier_params_optimized:
                #     print(
                #         '[ERROR] [CCCatheter] bezier curve parameters unavailable. Run calculate_beziers_control_points() first'
                #     )
                #     exit()
                
                # if use reconstruction result as feedback
                if hasattr(self, 'bezier_params_optimized'):
                    bezier_params_list = self.bezier_params_optimized
                    print("Use reconstruction result as feedback")
                # if use ground truth as feedback
                else:
                    bezier_params_list = self.bezier_params_list
                    print("Use ground truth as feedback")
                
                if self.tip_loss: # Bezier end control point error
                    self.p_diffs = np.zeros((3, 1))
                    self.p_diffs[:, 0] = self.target_bezier_params_list[-1] - bezier_params_list[-1]
                    print("Calculating 3D bezier tip loss")

                else: # Bezier end and middle control point error
                    self.p_diffs = np.zeros((3 * len(bezier_params_list), 1))

                    for i, (p, p_target) in enumerate(zip(bezier_params_list, self.target_bezier_params_list)):
                        self.p_diffs[i * 3:(i + 1) * 3, 0] = (p_target - p)
                    print("Calculating 3D bezier shape loss (using feedback from reconstruction)")
                    
    def calculate_loss(self):
        if self.loss_2d:
            if not self.target_cc_pt_list_2d:
                print('[ERROR] [CCCatheter] target_cc_pt_list_2d.')
                exit()
            if not self.bezier_params_list_2d:
                print('[ERROR] [CCCatheter] 2D bezier parameters invalid.')
                exit()
                
            if self.tip_loss:
                p_diffs = np.zeros((2, 1))
                p_diffs[:, 0] = self.target_cc_pt_list_2d[-1] - self.bezier_params_list_2d[-1]
                loss = np.linalg.norm(p_diffs)
                return loss

            else:
                p_diffs = np.zeros((2 * len(self.bezier_params_list_2d), 1))

                for i, (p, p_target) in enumerate(zip(self.bezier_params_list_2d, self.target_cc_pt_list_2d)):
                    p_diffs[i * 2:(i + 1) * 2, 0] = (p_target - p)
                loss = (np.linalg.norm(p_diffs[:2]) + np.linalg.norm(p_diffs[2:])) / 2
                return loss

        else:
            if not self.target_bezier_params_list or not self.bezier_params_list:
                print(
                    '[ERROR] [CCCatheter] self.target_bezier_params_list invalid. Run calculate_beziers_control_points(target=True) first'
                )
                exit()
            
            if self.tip_loss: # Bezier end control point error
                p_diffs = np.zeros((3, 1))
                p_diffs[:, 0] = self.target_bezier_params_list[-1] - self.bezier_params_list[-1]
                loss = np.linalg.norm(p_diffs)
                return loss

            else: # Bezier end and middle control point error
                p_diffs = np.zeros((3 * len(self.bezier_params_list), 1))

                for i, (p, p_target) in enumerate(zip(self.bezier_params_list, self.target_bezier_params_list)):
                    p_diffs[i * 3:(i + 1) * 3, 0] = (p_target - p)
                loss = (np.linalg.norm(p_diffs[:3]) + np.linalg.norm(p_diffs[3:])) / 2
                return loss
                    
    # def generate_noise(self, point, noise_level):
    #     """
    #     Generate noise based on the input point and a seed value.
    #     This ensures that similar points will generate similar noise.
    #     """
    #     seed = hash(tuple(point)) % 2**32
    #     np.random.seed(seed)
    #     noise = noise_level * np.random.randn(*point.shape)
    #     print("Noise generated: ", noise)
    #     return noise
    
    # # 生成用于平移的向量，确保相近的参考点产生相似的偏移向量
    # def generate_noise(self, reference_point, translation_magnitude=0.002, noise_scale=2.0):
    #     # 将参考点乘以一个缩放因子，确保平滑性
    #     scaled_point = reference_point / noise_scale
        
    #     # print(type(scaled_point), scaled_point)
    #     scaled_point = np.array(scaled_point, dtype=np.float64)
        
    #     # 使用参考点的坐标生成噪声方向向量，使用 sin 生成平滑随机值
    #     noise = np.sin(scaled_point * np.pi * 2)
        
    #     # 随机生成的方向不是单位向量，先归一化成单位向量
    #     random_direction = noise / np.linalg.norm(noise)
        
    #     # 将单位向量乘以 translation_magnitude，保证平移向量的模长为指定值
    #     translation_vector = random_direction * translation_magnitude
    #     print("Noise generated: ", translation_vector, "magnitude: ", np.linalg.norm(translation_vector))
    #     return translation_vector
    
    def generate_noise(self, reference_point, min_translation_magnitude=0.001, max_translation_magnitude=0.002, noise_scale=1.0, magnitude_variation_factor=20.0):
        """
        Generate a translation vector to account for noise
        Ensure that nearby reference points produce similar translation vectors
        """
        # Scale the reference point to ensure smooth noise generation
        scaled_point = reference_point / noise_scale
        
        # Ensure that scaled_point is of type float64 for consistency
        scaled_point = np.array(scaled_point, dtype=np.float64)
        
        # Generate noise based on the coordinates of the scaled point, using sin to generate smooth random values
        noise = np.sin(scaled_point * np.pi * 2)
        
        # Normalize the generated noise to create a unit direction vector
        random_direction = noise / np.linalg.norm(noise)
        
        # Generate a value between 0 and 1 based on smooth noise function
        scaled_magnitude = (np.sin(np.sum(scaled_point * magnitude_variation_factor)) + 1) / 2
        
        # Map the value from [0, 1] to [min_translation_magnitude, max_translation_magnitude]
        translation_magnitude = min_translation_magnitude + (max_translation_magnitude - min_translation_magnitude) * scaled_magnitude
        
        # Multiply the unit direction vector by the random translation magnitude to get the final translation vector
        translation_vector = random_direction * translation_magnitude
        
        print("Noise generated: ", translation_vector, "magnitude: ", np.linalg.norm(translation_vector))
        return translation_vector


    def calculate_beziers_control_points(self, noise_level=0, target=False):
        """
        Given the list of points on the constant curvature curve, calculate the control points for Bezier curves.
        Results: 
            4 Bezier control points (number of cc points // 2), for rendering with Blender.
            self.bezier_params_list: 3 Bezier control points, for reconstruction and control loop (ground truth).
            self.target_bezier_params_list: target
            self.bezier_params_list_theory: theoretical, without consideration of noise (user-capable data)
        Format: [b_mid, b_end], b_mid length = 3, b_end length = 3
        """
        
        # ------ Calculate the target Bezier control points ------
        if target:
            if not self.target_cc_pt_list:
                print('[ERROR] [CCCatheter] self.target_cc_pt_list invalid. Run calculate_cc_points(target=True) first')
                exit()
                
            cc_pt_list_with_p_0 = [self.p_0] + self.target_cc_pt_list

            n_beziers = int(len(self.cc_pt_list) / 2)
            for i in range(n_beziers):
                p_start = cc_pt_list_with_p_0[2 * i]
                p_mid = cc_pt_list_with_p_0[2 * i + 1]
                p_end = cc_pt_list_with_p_0[2 * (i + 1)]
                
                b1 = (p_mid - (self.p_0 / 4) - (p_end / 4)) * 2
                
                self.target_bezier_params_list = []
                self.target_bezier_params_list.append(b1)
                self.target_bezier_params_list.append(p_end)
                
            return
        
        # ------ Calculate the Bezier control points (real) ------
        if not self.cc_pt_list:
            print('[ERROR] [CCCatheter] self.cc_pt_list invalid. Run calculate_cc_points() first')
            exit()

        n_beziers = int(len(self.cc_pt_list) / 2)
        self.bezier_set = BezierSet(n_beziers)

        # Get the start, middle and end points of the constant curvature curve
        cc_pt_list_with_p_0 = [self.p_0] + self.cc_pt_list

        p_start = cc_pt_list_with_p_0[0]
        p_mid = cc_pt_list_with_p_0[1]
        p_end = cc_pt_list_with_p_0[2]
        
        self.bezier_params_list = []
        
        # Add noise to the conversion between constant curvature and Bezier control points
        # (Add noise to the middle and end points of the constant curvature curve)
        if noise_level > 0:
            # Generate deterministic noise based on the point coordinates
            noise_vector_mid = self.generate_noise(p_mid, max_translation_magnitude=noise_level)
            p_mid = p_mid + noise_vector_mid
            noise_vector_end = self.generate_noise(p_end, max_translation_magnitude=noise_level)
            p_end = p_end + noise_vector_end
            
            self.noise_list.append([np.linalg.norm(noise_vector_mid), np.linalg.norm(noise_vector_end)])
                    
        c = (p_mid - (self.p_0 / 4) - (p_end / 4)) * 2
        c1 = 4 / 3 * p_mid - 1 / 3 * p_end
        c2 = 4 / 3 * p_mid - 1 / 3 * p_start
        
        # 4 bezier control points, for rendering catheter with Blender
        self.bezier_set.enter_spec(p_start, p_end, c1, c2)
        
        # 3 bezier control points, for control loop
        self.bezier_params_list.append(c)
        self.bezier_params_list.append(p_end)

        if self.verbose > 1:
            print('Bezier ' + str(i) + ': ')
            print('    p_start = ', p_start)
            print('    p_mid   = ', p_mid)
            print('    p_end   = ', p_end)
        
        
        # ------ Calculate the Bezier control points (theoretical) ------
        if not self.cc_pt_list_theory:
            print('[ERROR] [CCCatheter] self.cc_pt_list_theory invalid. Run calculate_cc_points() first')
            exit()

        # Get the start, middle and end points of the constant curvature curve
        cc_pt_list_with_p_0 = [self.p_0] + self.cc_pt_list_theory

        p_start = cc_pt_list_with_p_0[0]
        p_mid = cc_pt_list_with_p_0[1]
        p_end = cc_pt_list_with_p_0[2]
                
        self.bezier_params_list_theory = []
         
        c = (p_mid - (self.p_0 / 4) - (p_end / 4)) * 2
        self.bezier_params_list_theory.append(c)
        self.bezier_params_list_theory.append(p_end)

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

        # self.bezier_set.print_specs()
        self.bezier_set.write_specs(curve_specs_path)
        self.bezier_set.render(img_save_path, target_specs_path, viewpoint_mode, transparent_mode)
        print(f"Image rendered. Image saved to {img_save_path}")

    def visualize_targets(self, img_save_path):
        """
        Visualize 2D target positions

        Args:
            img_save_path (path string to png file): path to save the visualization image
        """
        img = np.ones((480, 640, 3)) * 255

        for p in self.target_cc_pt_list_2d:
            x = int(p[0])
            y = int(p[1])

            print('target: x = ', x, '  y = ', y)

            img = cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        for p in self.cc_pt_list_2d:
            x = int(p[0])
            y = int(p[1])

            print('current: x = ', x, '  y = ', y)
            img = cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

        cv2.imwrite(img_save_path, img)

    def calculate_jacobian_1dof_3d(self):
        """
        Calculate the Jacobian for 1DoF control with 3D loss (1DoF is not fully implemented)
        """
        if self.tip_loss:
            J = np.zeros((3, 1))
            G_u = transforms.d_u_cc_transform_1dof(self.p_0, self.phi, self.u, self.l, self.r, s=1)
            J[:, 0] = G_u

        else:
            J = np.zeros((3 * len(self.cc_pt_list), 1))

            for i, s in enumerate(self.s_list):
                G_u = transforms.d_u_cc_transform_1dof(self.p_0, self.phi, self.u, self.l, self.r, s)
                J[i * 3:(i + 1) * 3, 0] = G_u

        return J

    def calculate_jacobian_1dof_2d(self):
        """
        Calculate the Jacobian for 1DoF control with 2D loss (1DoF is not fully implemented)
        """
        if not self.loss_2d:
            print('[ERROR] [CCCatheter] Not initialized with 2D Loss')
            exit()

        if self.tip_loss:
            J = np.zeros((3, 1))
            G_u = transforms.d_u_cc_transform_1dof(self.p_0, self.phi, self.u, self.l, self.r, s=1)
            J[:, 0] = G_u

            L_diag = transforms.world_to_image_interaction_matrix(self.cc_pt_list[-1], self.camera_extrinsics, self.fx,
                                                                  self.fy)

        else:
            J = np.zeros((3 * len(self.cc_pt_list), 1))
            L_diag = np.zeros((2 * len(self.cc_pt_list), 3 * len(self.cc_pt_list)))

            for i, (s, p) in enumerate(zip(self.s_list, self.cc_pt_list)):
                G_u = transforms.d_u_cc_transform_1dof(self.p_0, self.phi, self.u, self.l, self.r, s)
                J[i * 3:(i + 1) * 3, 0] = G_u

                L = transforms.world_to_image_interaction_matrix(p, self.camera_extrinsics, self.fx, self.fy)
                L_diag[i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = L

        return L_diag @ J

    def calculate_jacobian_2dof_3d(self):
        """
        Calculate the Jacobian for 2DoF control with 3D loss
        """
        if self.tip_loss:
            J = np.zeros((3, 2))

            G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)
            G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)

            J[:, 0] = G_ux
            J[:, 1] = G_uy

        else:
            J = np.zeros((3 * len(self.cc_pt_list), 2))

            for i, s in enumerate(self.s_list):
                G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)
                G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)

                J[i * 3:(i + 1) * 3, 0] = G_ux
                J[i * 3:(i + 1) * 3, 1] = G_uy

        return J

    def calculate_jacobian_2dof_2d(self):
        """
        Calculate the Jacobian for 2DoF control with 2D loss
        """
        if not self.loss_2d:
            print('[ERROR] [CCCatheter] Not initialized with 2D Loss')
            exit()

        if self.tip_loss:
            J = np.zeros((3, 2))

            G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)
            G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)

            J[:, 0] = G_ux
            J[:, 1] = G_uy

            L_diag = transforms.world_to_image_interaction_matrix(self.cc_pt_list[-1], self.camera_extrinsics, self.fx,
                                                                  self.fy)

        else:
            J = np.zeros((3 * len(self.cc_pt_list), 2))
            L_diag = np.zeros((2 * len(self.cc_pt_list), 3 * len(self.cc_pt_list)))

            for i, (s, p) in enumerate(zip(self.s_list, self.cc_pt_list)):
                G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)
                G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)

                J[i * 3:(i + 1) * 3, 0] = G_ux
                J[i * 3:(i + 1) * 3, 1] = G_uy

                L = transforms.world_to_image_interaction_matrix(p, self.camera_extrinsics, self.fx, self.fy)
                L_diag[i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = L

        return L_diag @ J

    def calculate_jacobian_3dof_3d(self):
        """
        Calculate the Jacobian for 3DoF control with 3D loss
        """
        if self.tip_loss:
            J = np.zeros((3, 3))

            G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)
            G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)
            G_l = transforms.d_l_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)

            J[:, 0] = G_ux
            J[:, 1] = G_uy
            J[:, 2] = G_l

        else:
            J = np.zeros((3 * len(self.cc_pt_list), 3))

            for i, s in enumerate(self.s_list):
                G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)
                G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)
                G_l = transforms.d_l_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)

                J[i * 3:(i + 1) * 3, 0] = G_ux
                J[i * 3:(i + 1) * 3, 1] = G_uy
                J[i * 3:(i + 1) * 3, 2] = G_l

        return J

    def calculate_jacobian_3dof_2d(self):
        """
        Calculate the Jacobian for 3DoF control with 2D loss
        """
        if not self.loss_2d:
            print('[ERROR] [CCCatheter] Not initialized with 2D Loss')
            exit()

        if self.tip_loss:
            J = np.zeros((3, 3))

            G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)
            G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)
            G_l = transforms.d_l_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s=1)

            J[:, 0] = G_ux
            J[:, 1] = G_uy
            J[:, 2] = G_l

            L_diag = transforms.world_to_image_interaction_matrix(self.cc_pt_list[-1], self.camera_extrinsics, self.fx,
                                                                  self.fy)

        else:
            J = np.zeros((3 * len(self.cc_pt_list), 3))
            L_diag = np.zeros((2 * len(self.cc_pt_list), 3 * len(self.cc_pt_list)))

            for i, (s, p) in enumerate(zip(self.s_list, self.cc_pt_list)):
                G_ux = transforms.d_ux_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)
                G_uy = transforms.d_uy_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)
                G_l = transforms.d_l_cc_transform_3dof(self.p_0, self.ux, self.uy, self.l, self.r, s)

                J[i * 3:(i + 1) * 3, 0] = G_ux
                J[i * 3:(i + 1) * 3, 1] = G_uy
                J[i * 3:(i + 1) * 3, 2] = G_l

                L = transforms.world_to_image_interaction_matrix(p, self.camera_extrinsics, self.fx, self.fy)
                L_diag[i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = L

        return L_diag @ J

    def update_1dof_params(self, current_iter, noise_percentage=0):
        """
        1DoF control in unispace (1DoF not fully implemented)
        
        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
        """
        self.calculate_p_diffs()
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        if self.loss_2d:
            J = self.calculate_jacobian_1dof_2d()
        else:
            J = self.calculate_jacobian_1dof_3d()

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix[0, 0]
        print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_u = d[0, 0]

        self.u += d_u

        if self.verbose > 0:
            print('d_u = ', d_u)
            print('Updated u = ', self.u)

    def update_2dof_params(self, current_iter, noise_percentage=0):
        """
        2DoF control in unispace
        
        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
        """
        self.calculate_p_diffs()
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        if self.loss_2d:
            J = self.calculate_jacobian_2dof_2d()
        else:
            J = self.calculate_jacobian_2dof_3d()

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix[:2, :2]
        print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_ux = d[0, 0]
        d_uy = d[1, 0]

        ## Add noise to parameter updates
        if noise_percentage > 0:
            d_ux = random.gauss(d_ux, noise_percentage * d_ux)
            d_uy = random.gauss(d_uy, noise_percentage * d_uy)

        ux_old = self.ux
        uy_old = self.uy

        self.ux += d_ux
        self.uy += d_uy

        ## View breach prevention
        self.calculate_cc_points(-1)
        while not self.convert_cc_points_to_2d(-1):
            print('[WARNING] View breach caught')

            d_ux /= 2
            d_uy /= 2

            self.ux = ux_old + d_ux
            self.uy = uy_old + d_uy
            self.calculate_cc_points(-1)

        self.params[current_iter + 1, 0] = self.ux
        self.params[current_iter + 1, 1] = self.uy

        if self.verbose > 0:
            print('d_ux = ', d_ux)
            print('d_uy = ', d_uy)
            print('Updated ux = ', self.ux)
            print('Updated uy = ', self.uy)

    def update_3dof_params(self, current_iter, noise_percentage=0):
        """
        3DoF control in unispace
        
        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
        """
        self.calculate_p_diffs()
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        if self.loss_2d:
            J = self.calculate_jacobian_3dof_2d()
        else:
            J = self.calculate_jacobian_3dof_3d()

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix
        print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_ux = d[0, 0]
        d_uy = d[1, 0]
        d_l = d[2, 0]

        ## Add noise to parameter updates
        if noise_percentage > 0:
            d_ux = random.gauss(d_ux, noise_percentage * d_ux)
            d_uy = random.gauss(d_uy, noise_percentage * d_uy)
            d_l = random.gauss(d_l, noise_percentage * d_l)

        ux_old = self.ux
        uy_old = self.uy
        l_old = self.l

        self.ux += d_ux
        self.uy += d_uy
        self.l += d_l

        ## View breach prevention
        self.calculate_cc_points(-1)
        while not self.convert_cc_points_to_2d(-1):
            print('[WARNING] View breach caught')

            d_ux /= 2
            d_uy /= 2
            d_l /= 2

            self.ux = ux_old + d_ux
            self.uy = uy_old + d_uy
            self.l = l_old + d_l
            self.calculate_cc_points(-1)

        self.params[current_iter + 1, 0] = self.ux
        self.params[current_iter + 1, 1] = self.uy
        self.params[current_iter + 1, 2] = self.l

        if self.verbose > 0:
            print('d_ux = ', d_ux)
            print('d_uy = ', d_uy)
            print('d_l  = ', d_l)
            print('Updated ux = ', self.ux)
            print('Updated uy = ', self.uy)
            print('Updated l  = ', self.l)

    def update_2dof_params_bezier_interspace_ux_uy(self, current_iter, noise_percentage=0):
        """
        2DoF control in Bezier interspace with (ux, uy) parameterization
        
        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
        """
        print('Running control of 2-DOF Interspace (ux, uy) Parameterization')

        self.calculate_p_diffs(bezier=True)
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        J = bezier_interspace_transforms.calculate_jacobian_2dof_ux_uy(self.p_0, self.ux, self.uy, self.l, self.r)

        if self.tip_loss:
            J = J[-3:, :]

        if self.loss_2d:

            if self.tip_loss:
                # L_diag = transforms.world_to_image_interaction_matrix(self.cc_pt_list[-1], self.camera_extrinsics,
                #                                                       self.fx, self.fy)
                
                # if use reconstruction result as feedback
                if hasattr(self, 'bezier_params_optimized'):
                    bezier_params_list = self.bezier_params_optimized   
                # if use ground truth as feedback
                else:
                    bezier_params_list = self.bezier_params_list
                    
                L_diag = transforms.world_to_image_interaction_matrix(bezier_params_list[-1], self.camera_extrinsics,
                                                                      self.fx, self.fy)

            else:
                L_diag = np.zeros((2 * len(self.cc_pt_list), 3 * len(self.cc_pt_list)))

                for i, p in enumerate(self.cc_pt_list):
                    L = transforms.world_to_image_interaction_matrix(p, self.camera_extrinsics, self.fx, self.fy)
                    L_diag[i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = L

            J = L_diag @ J

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix[:2, :2]
        # print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_ux = d[0, 0]
        d_uy = d[1, 0]

        self.du = [d_ux, d_uy]
        self.ux_theory += d_ux
        self.uy_theory += d_uy
        
        ## Add noise to parameter updates
        if noise_percentage > 0:
            d_ux = random.gauss(d_ux, noise_percentage * d_ux)
            d_uy = random.gauss(d_uy, noise_percentage * d_uy)

        ux_old = self.ux
        uy_old = self.uy

        self.ux += d_ux
        self.uy += d_uy

        # Check if the catheter tip is out of image view
        self.calculate_cc_points(-1)
        while not self.convert_cc_points_to_2d(-1):
            print('[WARNING] View breach caught')

            d_ux /= 2
            d_uy /= 2

            self.ux = ux_old + d_ux
            self.uy = uy_old + d_uy
            self.calculate_cc_points(-1)

        self.params[current_iter + 1, 0] = self.ux
        self.params[current_iter + 1, 1] = self.uy

        if self.verbose > 0:
            print('d_ux = ', d_ux)
            print('d_uy = ', d_uy)
            print('Updated ux = ', self.ux)
            print('Updated uy = ', self.uy)

    def update_3dof_params_bezier_interspace_ux_uy(self, current_iter, noise_percentage=0):
        """
        3DoF control in Bezier interspace with (ux, uy) parameterization
        
        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
        """
        print('Running control of 3-DOF Interspace (ux, uy) Parameterization')

        self.calculate_p_diffs(bezier=True)
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        J = bezier_interspace_transforms.calculate_jacobian_3dof_ux_uy(self.p_0, self.ux, self.uy, self.l, self.r)

        if self.tip_loss:
            J = J[-3:, :]

        if self.loss_2d:

            if self.tip_loss:
                # if use reconstruction result as feedback
                if hasattr(self, 'bezier_params_optimized'):
                    bezier_params_list = self.bezier_params_optimized   
                # if use ground truth as feedback
                else:
                    bezier_params_list = self.bezier_params_list
                    
                L_diag = transforms.world_to_image_interaction_matrix(bezier_params_list[-1], self.camera_extrinsics,
                                                                      self.fx, self.fy)

            else:
                L_diag = np.zeros((2 * len(self.cc_pt_list), 3 * len(self.cc_pt_list)))

                for i, p in enumerate(self.cc_pt_list):
                    L = transforms.world_to_image_interaction_matrix(p, self.camera_extrinsics, self.fx, self.fy)
                    L_diag[i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = L

            J = L_diag @ J

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix
        # print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_ux = d[0, 0]
        d_uy = d[1, 0]
        d_l = d[2, 0]
        
        self.du = [d_ux, d_uy]
        self.ux_theory += d_ux
        self.uy_theory += d_uy
        
        ## Add noise to parameter updates
        if noise_percentage > 0:
            d_ux = random.gauss(d_ux, noise_percentage * d_ux)
            d_uy = random.gauss(d_uy, noise_percentage * d_uy)
        
        ux_old = self.ux
        uy_old = self.uy
        l_old = self.l

        self.ux += d_ux
        self.uy += d_uy
        self.l += d_l

        # Check if the catheter tip is out of image view
        self.calculate_cc_points(-1)
        while not self.convert_cc_points_to_2d(-1):
            print('[WARNING] View breach caught')

            d_ux /= 2
            d_uy /= 2
            d_l /= 2

            self.ux = ux_old + d_ux
            self.uy = uy_old + d_uy
            self.l = l_old + d_l
            self.calculate_cc_points(-1)

        self.params[current_iter + 1, 0] = self.ux
        self.params[current_iter + 1, 1] = self.uy
        self.params[current_iter + 1, 2] = self.l
        
        print("l = ", self.l)

        if self.verbose > 0:
            print('d_ux = ', d_ux)
            print('d_uy = ', d_uy)
            print('d_l  = ', d_l)
            print('Updated ux = ', self.ux)
            print('Updated uy = ', self.uy)
            print('Updated l  = ', self.l)

    def update_2dof_params_bezier_interspace_theta_phi(self, current_iter, noise_percentage=0):
        """
        2DoF control in Bezier interspace with (theta, phi) parameterization
        
        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
        """
        print('Running 2-DOF Interspace (theta, phi) Parameterization')

        self.calculate_p_diffs()
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        theta = np.sqrt(self.ux**2 + self.uy**2) / self.r
        phi = np.arctan2(self.uy, self.ux)

        if current_iter == 0:
            self.params[0, 3] = theta
            self.params[0, 4] = phi

            try:
                theta_target = np.sqrt(self.ux_target**2 + self.uy_target**2) / self.r
                phi_target = np.arctan2(self.uy_target, self.ux_target)

                self.params[-1, 3] = theta_target
                self.params[-1, 4] = phi_target
            except:
                print('[Warning] Writing target theta and phi params unsuccessful')

        J = bezier_interspace_transforms.calculate_jacobian_2dof_theta_phi(self.p_0, theta, phi, self.l, self.r)

        if self.tip_loss:
            J = J[-3:, :]

        if self.loss_2d:

            if self.tip_loss:
                L_diag = transforms.world_to_image_interaction_matrix(self.cc_pt_list[-1], self.camera_extrinsics,
                                                                      self.fx, self.fy)

            else:
                L_diag = np.zeros((2 * len(self.cc_pt_list), 3 * len(self.cc_pt_list)))

                for i, p in enumerate(self.cc_pt_list):
                    L = transforms.world_to_image_interaction_matrix(p, self.camera_extrinsics, self.fx, self.fy)
                    L_diag[i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = L

            J = L_diag @ J

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix[:2, :2]
        print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_theta = d[0, 0]
        d_phi = d[1, 0]

        ## Add noise to parameter updates
        if noise_percentage > 0:
            d_theta = random.gauss(d_theta, noise_percentage * d_theta)
            d_phi = random.gauss(d_phi, noise_percentage * d_phi)

        theta_old = theta
        phi_old = phi

        theta += d_theta
        phi += d_phi

        self.ux = bezier_interspace_transforms.calculate_ux(theta, phi, self.r)
        self.uy = bezier_interspace_transforms.calculate_uy(theta, phi, self.r)

        ## View breach prevention
        self.calculate_cc_points(-1)
        while not self.convert_cc_points_to_2d(-1):
            print('[WARNING] View breach caught')

            d_theta /= 2
            d_phi /= 2

            theta = theta_old + d_theta
            phi = phi_old + d_phi

            self.ux = bezier_interspace_transforms.calculate_ux(theta, phi, self.r)
            self.uy = bezier_interspace_transforms.calculate_uy(theta, phi, self.r)
            self.calculate_cc_points(-1)

        self.params[current_iter + 1, 0] = self.ux
        self.params[current_iter + 1, 1] = self.uy
        self.params[current_iter + 1, 3] = theta
        self.params[current_iter + 1, 4] = phi

        if self.verbose > 0:
            print('d_theta = ', d_theta)
            print('d_phi = ', d_phi)
            print('Updated ux = ', self.ux)
            print('Updated uy = ', self.uy)

    def update_3dof_params_bezier_interspace_theta_phi(self, current_iter, noise_percentage=0):
        """
        3DoF control in Bezier interspace with (theta, phi) parameterization
        
        Args:
            current_iter (int): current iteration in optimization.
                This is only used for parameter recording
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
        """
        print('Running 3-DOF Interspace (theta, phi) Parameterization')

        self.calculate_p_diffs()
        print('|p_diffs| = ', np.linalg.norm(self.p_diffs))

        theta = np.sqrt(self.ux**2 + self.uy**2) / self.r
        phi = np.arctan2(self.uy, self.ux)

        if current_iter == 0:
            self.params[0, 3] = theta
            self.params[0, 4] = phi

            try:
                theta_target = np.sqrt(self.ux_target**2 + self.uy_target**2) / self.r
                phi_target = np.arctan2(self.uy_target, self.ux_target)

                self.params[-1, 3] = theta_target
                self.params[-1, 4] = phi_target
            except:
                print('[Warning] Writing target theta and phi params unsuccessful')

        J = bezier_interspace_transforms.calculate_jacobian_3dof_theta_phi(self.p_0, theta, phi, self.l, self.r)

        if self.tip_loss:
            J = J[-3:, :]

        if self.loss_2d:

            if self.tip_loss:
                L_diag = transforms.world_to_image_interaction_matrix(self.cc_pt_list[-1], self.camera_extrinsics,
                                                                      self.fx, self.fy)

            else:
                L_diag = np.zeros((2 * len(self.cc_pt_list), 3 * len(self.cc_pt_list)))

                for i, p in enumerate(self.cc_pt_list):
                    L = transforms.world_to_image_interaction_matrix(p, self.camera_extrinsics, self.fx, self.fy)
                    L_diag[i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = L

            J = L_diag @ J

        J_T = np.transpose(J)

        weight_matrix = self.weight_matrix
        print('weight_matrix = ', weight_matrix)

        d = np.linalg.pinv(J_T @ J + weight_matrix) @ J_T @ self.p_diffs
        d_theta = d[0, 0]
        d_phi = d[1, 0]
        d_l = d[2, 0]

        ## Add noise to parameter updates
        if noise_percentage > 0:
            d_theta = random.gauss(d_theta, noise_percentage * d_theta)
            d_phi = random.gauss(d_phi, noise_percentage * d_phi)
            d_l = random.gauss(d_l, noise_percentage * d_l)

        theta_old = theta
        phi_old = phi
        l_old = self.l

        theta += d_theta
        phi += d_phi

        self.ux = bezier_interspace_transforms.calculate_ux(theta, phi, self.r)
        self.uy = bezier_interspace_transforms.calculate_uy(theta, phi, self.r)
        self.l += d_l

        ## View breach prevention
        self.calculate_cc_points(-1)
        while not self.convert_cc_points_to_2d(-1):
            print('[WARNING] View breach caught')

            d_theta /= 2
            d_phi /= 2
            d_l /= 2

            theta = theta_old + d_theta
            phi = phi_old + d_phi

            self.ux = bezier_interspace_transforms.calculate_ux(theta, phi, self.r)
            self.uy = bezier_interspace_transforms.calculate_uy(theta, phi, self.r)
            self.l = l_old + d_l
            self.calculate_cc_points(-1)

        self.params[current_iter + 1, 0] = self.ux
        self.params[current_iter + 1, 1] = self.uy
        self.params[current_iter + 1, 2] = self.l
        self.params[current_iter + 1, 3] = theta
        self.params[current_iter + 1, 4] = phi

        if self.verbose > 0:
            print('d_theta = ', d_theta)
            print('d_phi = ', d_phi)
            print('d_l  = ', d_l)
            print('Updated ux = ', self.ux)
            print('Updated uy = ', self.uy)
            print('Updated l  = ', self.l)
