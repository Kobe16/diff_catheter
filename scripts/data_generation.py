import random
import numpy as np
import transforms
import os

from cc_catheter import CCCatheter


class DataGeneration:

    def __init__(self, n_data, p_0, r, l, s_list, save_path):
        """
        Args:
            n_data (int): number of data points (variations of parameters as test cases) to generate
            p_0 ((3,) numpy array): start point of catheter
            r (float): cross section radius of catheter
            l (float): length of catheter
            s_list (list of s values): s values of points on the constant curvature curve except for
                the start point. s values are floats from 0 to 1 inclusive
            save_path (path string to npy file): path to save the generated data
        """
        self.n_data = n_data
        self.p_0 = p_0
        self.r = r
        self.l = l
        self.save_path = save_path
        self.s_list = s_list

        random.seed(0)

    
    def set_target_ranges(self, ux_min, ux_max, uy_min, uy_max, l_min, l_max):
        """
        Args:
            ux_min (float): minimum value of ux to be generated in data
            ux_max (float): maximum value of ux to be generated in data
            uy_min (float): minimum value of uy to be generated in data
            uy_max (float): maximum value of uy to be generated in data
            l_min (float): minimum value of l to be generated in data
            l_max (float): maximum value of l to be generated in data
        """
        self.ux_min = ux_min
        self.ux_max = ux_max
        self.uy_min = uy_min
        self.uy_max = uy_max
        self.l_min = l_min
        self.l_max = l_max


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


    def check_view_boundary(self, ux_target, uy_target, l_target):
        """
        Check whether the given targets result in the end point of catheter falling outside of camera view
        
        Args:
            ux_target (float): 1st pair of tendon length (responsible for catheter bending)
            uy_target (float): 2nd pair of tendon length (responsible for catheter bending)
            l_target (float): length of bending portion of the catheter (responsible for insertion)

        Returns:
            (bool): whether the given targets result in the end point of catheter falling outside
                of camera view
        """
        for s in self.s_list:

            p_3d = transforms.cc_transform_3dof(self.p_0, ux_target, uy_target, l_target, self.r, s)
            p_2d = transforms.world_to_image_transform(p_3d, self.camera_extrinsics, self.fx, self.fy, self.cx, self.cy)

            p_2d[0] = round(self.size_x - p_2d[0])
            p_2d[1] = round(p_2d[1])

            margin = 10 # in pixels
            #if p_2d[0] >= self.size_x or p_2d[0] < 0 or p_2d[1] >= self.size_y or p_2d[1] < 0:  
            if p_2d[0] >= self.size_x - margin or p_2d[0] < margin or p_2d[1] >= self.size_y - margin or p_2d[1] < margin:                
                return False

        return True


    def generate_random_float(self, range_min, range_max):
        """
        Ouputs a random number within the given range
        """
        return random.random() * (range_max - range_min) + range_min


    def generate_data(self):
        """
        Generate a certain number of data within the specified parameter ranges and
            save the data in a given path 
        """
        print("Start generating data for targets")
        self.generated_data = np.zeros((self.n_data, 3))

        n_iter = 0
        n_valid_data = 0

        while n_valid_data < self.n_data:

            # print('n_iter = ', n_iter, ' n_valid_data = ', n_valid_data)

            ux_target = self.generate_random_float(self.ux_min, self.ux_max)
            uy_target = self.generate_random_float(self.uy_min, self.uy_max)
            l_target = self.generate_random_float(self.l_min, self.l_max)

            ## Boundary with the generated l_target and the initial l must both be checked to accomodate both 2-DOF and 3-DOF experiments
            if self.check_view_boundary(ux_target, uy_target, l_target) and self.check_view_boundary(ux_target, uy_target, self.l):
                self.generated_data[n_valid_data, :] = np.array([ux_target, uy_target, l_target])
                n_valid_data += 1

            n_iter += 1
        
        print('Generated ', n_valid_data, ' data in ', n_iter, ' iterations')
        # print(self.generated_data)

        np.save(self.save_path, self.generated_data)
        
    def visualize_targets(self, save_path, n_mid_points, n_iter):
        for i, targets in enumerate(self.generated_data):
            ux = targets[0]
            uy = targets[1]
            l = targets[2]
            
            catheter = CCCatheter(self.p_0, self.l, self.r, None, None, n_mid_points, n_iter, verbose=0)
            catheter.set_3dof_params(ux, uy, l)
            catheter.calculate_cc_points()
            catheter.calculate_beziers_control_points()
            
            name = f'target_{i}'
            curve_specs_path = os.path.join(save_path , name + '.npy')
            image_save_path = os.path.join(save_path, name + '.png')
            
            print('Rendering image of target bezier curve')
            catheter.render_beziers(curve_specs_path, image_save_path)
