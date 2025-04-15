import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

import camera_settings
import path_settings
from cc_catheter import CCCatheter
# from reconstruction_scripts.reconst_sim_opt2pts import reconstructCurve
from catheter_reconstruction.reconst_2_loss import optimize
from catheter_reconstruction.reconst_3_loss import optimize_3_loss
from catheter_reconstruction.utils import *

class SimulationExperiment:

    def __init__(self, dof, loss_2d, tip_loss, use_reconstruction, interspace, viewpoint_mode, damping_weights, n_iter, render_mode):
        """
        Args:
            dof (1, 2, or 3): DoF of control (1 DoF is not fully implemented currently)
            loss_2d (bool): whether to use 2D loss
            tip_loss (bool): whether to use tip loss
            use_recontruction (0, 1 or 2): 0 means full simulation instead of using reconstruction, 1 means using reconstruction 
            with 2 loss functions, 2 means using reconstruction with 3 loss functions 
            interspace (0, 1 or 2): interspace of control, 0 for unispace, 1 for Bezier interspace with (ux, uy) parameterization, 
                2 for Bezier interspace with (theta, phi) parameterization
            viewpoint_mode (1 or 2): camera view of rendered image, 1 for endoscopic view, 2 for side view
            damping_weights (list of 3 floats): n-th term corresponds to the damping weight of the n-th DoF control feedback
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
            n_iter (int): number of total iteration of optimization
            render_mode (0, 1, or 2): 0 for rendering no image, 1 for only rendering the image after
                the last iteration, 2 for rendering all image 
        """
        self.dof = dof
        self.loss_2d = loss_2d
        self.tip_loss = tip_loss
        # self.use_reconstruction = use_reconstruction
        self.interspace = interspace
        self.viewpoint_mode = viewpoint_mode
        self.damping_weights = damping_weights
        self.n_iter = n_iter
        self.render_mode = render_mode
        
        if use_reconstruction == 0:
            self.use_reconstruction = False
        if use_reconstruction == 1:
            self.use_reconstruction = True
            self.reconstruction_mode = 2
        else:
            self.use_reconstruction = True
            self.reconstruction_mode = 3
            

        self.use_2d_pos_target = False
        
        self.image_path_list = []
        self.delta_u_list = []
        self.u_list = []
        
        # self.diff_list = []
        self.loss_list = []
        self.bezier_list = []

    def set_paths(self, images_save_dir, cc_specs_save_dir, params_report_path, p3d_report_path, p2d_report_path):
        """
        Args:
            images_save_dir (path string to directory): directory to save rendered images
            cc_specs_save_dir (path string to directory): directory to save constant curvature specs
            params_report_path (path string to npy file): self.params is a (n_iter + 2, 5) numpy array
                the first row records the initial parameters;
                the last row records the target parameters;
                and the intermediate rows record the parameters throughout the iterations.
                The 5 columns records the ux, uy, l, theta, phi parameters.
                If some parameters are not applicable for current method, they are left as 0
            p3d_report_path (path string to npy file)
            p2d_report_path (path string to npy file)
        """
        self.images_save_dir = images_save_dir
        self.cc_specs_save_dir = cc_specs_save_dir
        self.params_report_path = params_report_path
        self.p3d_report_path = p3d_report_path        
        self.p2d_report_path = p2d_report_path
        
        
    def set_general_parameters(self, p_0, r, n_mid_points, l):
        """
        Args:
            p_0 ((3,) numpy array): start point of catheter
            r (float): cross section radius of catheter
            n_mid_points (int): number of middle control points
            l (float): length of catheter
        """
        self.p_0 = p_0
        self.r = r
        self.l = l
        self.n_mid_points = n_mid_points


    def set_1dof_parameters(self, u, phi, u_target):
        """
        Set parameters for 1DoF (not fully implemented currently)

        Args:
            u (float): tendon length (responsible for catheter bending)
            phi (radians as float): phi parameter
            u_target (float): target tendon length (responsible for catheter bending)
        """
        self.u = u
        self.phi = phi
        self.u_target = u_target


    def set_2dof_parameters(self, ux, uy, ux_target, uy_target):
        """
        Set parameters for 2DoF

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            ux_target (int): target of 1st pair of tendon length
            uy_target (int): target of 2nd pair of tendon length
        """
        self.ux = ux
        self.uy = uy
        self.ux_target = ux_target
        self.uy_target = uy_target


    def set_3dof_parameters(self, ux, uy, ux_target, uy_target, l_target):
        """
        Set parameters for 3DoF

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            ux_target (int): target of 1st pair of tendon length
            uy_target (int): target of 2nd pair of tendon length
            l_target (float): target length of bending portion of the catheter (responsible for insertion)
        """
        self.ux = ux
        self.uy = uy
        self.ux_target = ux_target
        self.uy_target = uy_target
        self.l_target = l_target

    
    def set_2d_pos_parameters(self, ux, uy, x_target, y_target, l=0):
        """
        Set parameters for 2D loss

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
            x_target (int): horizontal target pixel location of end effector
            y_target (int): vertical target pixel location of end effector
            l (float): length of bending portion of the catheter (responsible for insertion)
        """
        if not (self.loss_2d and self.tip_loss):
            print('[ERROR] Setting 2D position target is not compatible with non 2D tip loss')
            exit()

        self.ux = ux
        self.uy = uy
        self.x_target = x_target
        self.y_target = y_target

        if self.dof == 3:
            self.l = l

        self.use_2d_pos_target = True
        
    def set_noise_percentage(self, u_noise_percentage, cc_to_bezier_noise, feedback_u_noise_percentage):
        self.u_noise_percentage = u_noise_percentage
        self.cc_to_bezier_noise = cc_to_bezier_noise
        self.feedback_u_noise_percentage = feedback_u_noise_percentage

    # def plot_loss(self, show=False):
    #     """
    #     Plot the loss of each iteration
    #     """
    #     if not self.diff_list:
    #         raise ValueError("The loss_list is empty.")
            
    #     # Check if loss_list is empty
    #     if not self.loss_2d:
    #         ylabel = 'Error (mm)'
    #         if self.tip_loss:
    #             if self.diff_list[0].shape != (3, 1):
    #                 raise ValueError("Each element in loss_list should be a 3x1 vector when using tip loss.")
    #         else:
    #             if self.diff_list[0].shape != (6, 1):
    #                 raise ValueError("Each element in loss_list should be a 6x1 vector when using shape loss.")

    #     # Calculate the new loss list based on the value of tip_loss
    #     loss_list = []
    #     for diff in self.diff_list:
    #         if self.tip_loss:
    #             norm = np.linalg.norm(diff)
    #             loss_list.append(norm)
    #             title = '3D Tip Euclidean Distance Loss History'
    #         else:
    #             norm = (np.linalg.norm(diff[:3]) + np.linalg.norm(diff[3:])) / 2
    #             loss_list.append(norm)
    #             title = '3D Shape Loss History'
                
    #     # Plot the loss
    #     fig1 = plt.figure()
    #     ax1 = fig1.add_subplot(111)
    #     fig1.suptitle(title)
    #     ax1.plot(loss_list)
    #     ax1.set_xlabel('Iteration')
    #     ax1.set_ylabel(ylabel)
    #     ax1.set_xlim([0, len(loss_list)])
    #     # ax1.set_ylim([0, 80])
    #     ax1.set_ylim(bottom=0)
    #     ax1.grid(True)
        
    #     fig_path = os.path.join(self.images_save_dir, 'loss.png')
    #     if not os.path.exists(os.path.dirname(fig_path)):
    #         os.makedirs(os.path.dirname(fig_path))
    #     plt.savefig(fig_path)
    #     print('Plot of loss history saved at', fig_path)

    #     if show:
    #         plt.show()
            
    #     file_path = os.path.join(self.images_save_dir, 'loss.pkl')
    #     with open(file_path, 'wb') as file:
    #         pickle.dump(loss_list, file)
    
    def plot_loss(self, show=False, log_scale=False):
        """
        Plot the loss of each iteration
        """            
        # Check if loss_list is empty
        if self.loss_2d:
            ylabel = 'Error (pixels)'
            if self.tip_loss:
                title = '2D Tip Euclidean Distance Loss History'
            else:
                title = '2D Shape Loss History'
        
        else:
            ylabel = 'Error (mm)'
            if self.tip_loss:
                title = '3D Tip Euclidean Distance Loss History'
            else:
                title = '3D Shape Loss History'
                
        # Plot the loss
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig1.suptitle(title)
        ax1.plot(self.loss_list)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel(ylabel)
        ax1.set_xlim([0, len(self.loss_list)])
        # ax1.set_ylim([0, 80])
        if log_scale:
            # ax1.set_ylim(bottom=1)
            ax1.set_yscale('log')
            ax1.set_ylabel(ylabel + ' (log scale)')
        else:
            ax1.set_ylim(bottom=0)
        ax1.grid(True)
        
        fig_path = os.path.join(self.images_save_dir, 'loss.png')
        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))
        plt.savefig(fig_path)
        print('Plot of loss history saved at', fig_path)

        if show:
            plt.show()
            
        file_path = os.path.join(self.images_save_dir, 'loss.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(self.loss_list, file)

    def execute(self):
        """
        Run main pipeline of inverse Jacobian control

        Returns:
            params ((n_iter + 2, 5) numpy array): 
                the first row records the initial parameters;
                the last row records the target parameters;
                and the intermediate rows record the parameters throughout the iterations.
                The 5 columns records the ux, uy, l, theta, phi parameters.
                If some parameters are not applicable for current method, they are left as 0
        """
        
        # ----- Create a catheter class, initialize control parameters and target, camera parameters -----
        
        catheter = CCCatheter(self.p_0, self.l, self.r, self.loss_2d, self.tip_loss, self.n_mid_points, self.n_iter, verbose=0)
        catheter.set_weight_matrix(self.damping_weights[0], self.damping_weights[1], self.damping_weights[2])

        if self.use_2d_pos_target:
            assert self.ux
            assert self.uy
            assert self.x_target
            assert self.y_target

            if self.dof == 2:
                catheter.set_2dof_params(self.ux, self.uy)
            elif self.dof == 3:
                catheter.set_3dof_params(self.ux, self.uy, self.l)
            else:
                print('[ERROR] DOF invalid')

            catheter.set_2d_targets([0, self.x_target], [0, self.y_target])

        else:
            if self.dof == 1:
                assert self.u
                assert self.phi
                assert self.u_target

                catheter.set_1dof_params(self.phi, self.u)
                catheter.set_1dof_targets(self.u_target)

            elif self.dof == 2:
                assert self.ux
                assert self.uy
                assert self.ux_target
                assert self.uy_target

                catheter.set_2dof_params(self.ux, self.uy)
                catheter.set_2dof_targets(self.ux_target, self.uy_target)

            elif self.dof == 3:
                assert self.ux
                assert self.uy
                assert self.ux_target
                assert self.uy_target
                assert self.l_target

                catheter.set_3dof_params(self.ux, self.uy, self.l)
                catheter.set_3dof_targets(self.ux_target, self.uy_target, self.l_target)

            else:
                print('[ERROR] DOF invalid')

        catheter.set_camera_params(camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y, camera_settings.image_size_x, camera_settings.image_size_y, camera_settings.extrinsics)
        
        print("catheter parameters initialized")
        print('------------------------- Start of Simulation -------------------------')
        
        # ----- Apply motion model to simulate real catheter behavior (initial step) -----
        catheter.calculate_cc_points(init=True, noise_percentage=self.u_noise_percentage) # 0.05
        catheter.convert_cc_points_to_2d(init=True)
        catheter.calculate_beziers_control_points(noise_level=self.cc_to_bezier_noise) # 0.002
        catheter.convert_bezier_points_to_2d()
        
        self.u_list.append([catheter.ux, catheter.uy])
        self.bezier_list.append(np.vstack((self.p_0, catheter.bezier_params_list))) 

        if not self.use_2d_pos_target: # if use 3D target, set target specs
            catheter.calculate_cc_points(target=True)
            catheter.convert_cc_points_to_2d(target=True)
            catheter.calculate_beziers_control_points(target=True)
            
        self.loss_list.append(catheter.calculate_loss())
        print("Loss: ", self.loss_list[-1])
        
        cc_specs_path = os.path.join(self.cc_specs_save_dir , '000.npy')
        image_save_path = os.path.join(self.images_save_dir, '000.png')

        if self.use_reconstruction or self.use_2d_pos_target:
            self.bezier_specs_old = catheter.calculate_bezier_specs(init=True)
            target_specs_path = None
        else:
            target_specs_path = os.path.join(self.cc_specs_save_dir, 'target.npy')
            catheter.write_target_specs(target_specs_path, show_mid_points=True)

        # render image using Blender
        # if use reconstruction, self.render_mode == 2
        if self.render_mode == 2: 
            print('Rendering initial catheter image')
            catheter.render_beziers(cc_specs_path, image_save_path, target_specs_path, self.viewpoint_mode, transparent_mode=0)
            self.image_path_list.append(image_save_path)

        # ----- Start control loop -----
        
        for i in range(self.n_iter):
            print('------------------------- Start of Iteration ' + str(i) + ' -------------------------')
            
            # ----- Use reconstruction method to obtain catheter shape feedback from endoscopic image -----
            if self.use_reconstruction:
                
                # Get Bezier specs of current curve
                # bezier_specs = catheter.calculate_bezier_specs()
                # bezier_specs_torch = torch.tensor(bezier_specs.flatten(), dtype=torch.float)
                
                # Initial guess: the bezier specs of the previous iteration
                bezier_specs_init_torch = torch.tensor(self.bezier_specs_old.flatten(), dtype=torch.float)

                # loss_weight = torch.tensor([1.0, 1.0, 1.0])
                # p_0 = torch.tensor(catheter.p_0)
                
                # Apply reconstruction optimizer
                print("Begin reconstruction optimization ----------------------")
                if i == 0:
                    optimized_bezier = optimize(image_save_path, cc_specs_path, i, self.images_save_dir, bezier_specs_init_torch, learning_rate=4e-3)
                if self.reconstruction_mode == 2 and i > 0:
                    optimized_bezier = optimize(image_save_path, cc_specs_path, i, self.images_save_dir, bezier_specs_init_torch)
                if self.reconstruction_mode == 3 and i > 0:
                    if i == 1:
                        gt_img_path_list = self.image_path_list[:-1] # remove the lastest image
                        delta_u_list = self.delta_u_list
                        lr = 3e-3 # 3e-3

                    if i > 1:
                        # remove the lastest image and get the remaining last 2 images, then reverse the order
                        # gt_img_path_list = self.image_path_list[-3:-1][::-1] 
                        # delta_u_list = self.delta_u_list[-2:][::-1]
                        
                        gt_img_path_list = self.image_path_list[:2][::-1] 
                         
                        last_ux, last_uy = self.u_list[-1]
                        second_ux, second_uy = self.u_list[1]
                        delta_u_list = [[last_ux - second_ux, last_uy - second_uy], self.delta_u_list[0]]
                        
                        lr = 2e-3 # 2e-3, 5e-3
                    optimized_bezier = optimize_3_loss(image_save_path, cc_specs_path, i, self.images_save_dir, bezier_specs_init_torch, gt_img_path_list, delta_u_list, catheter.l, learning_rate=lr)
                # convert optimized bezier parameters to 3D and 2D contant curvature parameters
                optimized_bezier_specs = optimized_bezier.reshape((2, 3))
                catheter.write_bezier_specs(optimized_bezier_specs, self.use_reconstruction)
                catheter.convert_bezier_to_cc(optimized_bezier_specs)
                catheter.convert_cc_points_to_2d(i)

                # bezier_specs_old = bezier_specs
                self.bezier_specs_old = optimized_bezier_specs
                
                print("Reconstruction optimization finished ----------------------")

            if self.dof == 1:
                catheter.update_1dof_params(i, self.feedback_u_noise_percentage)

            elif self.dof == 2:
                if self.interspace == 0:
                    catheter.update_2dof_params(i, self.feedback_u_noise_percentage)
                elif self.interspace == 1:
                    catheter.update_2dof_params_bezier_interspace_ux_uy(i, self.feedback_u_noise_percentage)
                elif self.interspace == 2:
                    catheter.update_2dof_params_bezier_interspace_theta_phi(i, self.feedback_u_noise_percentage)

            else:
                if self.interspace == 0:
                    catheter.update_3dof_params(i, self.feedback_u_noise_percentage)
                elif self.interspace == 1:
                    catheter.update_3dof_params_bezier_interspace_ux_uy(i, self.feedback_u_noise_percentage)
                elif self.interspace == 2:
                    catheter.update_3dof_params_bezier_interspace_theta_phi(i, self.feedback_u_noise_percentage)

            # self.diff_list.append(catheter.p_diffs)
            
            self.delta_u_list.append(catheter.du)
            catheter.calculate_cc_points(i)
            catheter.convert_cc_points_to_2d(i)
            catheter.calculate_beziers_control_points(noise_level=self.cc_to_bezier_noise) # 0.007
            catheter.convert_bezier_points_to_2d()
            
            self.loss_list.append(catheter.calculate_loss())
            print("Loss: ", self.loss_list[-1]) 
            
            self.u_list.append([catheter.ux, catheter.uy])  
            self.bezier_list.append(np.vstack((self.p_0, catheter.bezier_params_list)))         

            cc_specs_path = os.path.join(self.cc_specs_save_dir, str(i + 1).zfill(3) + '.npy')
            image_save_path = os.path.join(self.images_save_dir, str(i + 1).zfill(3) + '.png')


            if self.render_mode > 0:
                if i == (self.n_iter - 1):
                    print("Rendering catheter image of iteration ", i)
                    catheter.render_beziers(cc_specs_path, image_save_path, target_specs_path, self.viewpoint_mode, transparent_mode=1)
                    self.image_path_list.append(image_save_path)
                elif self.render_mode == 2:
                    print("Rendering catheter image of iteration ", i)
                    catheter.render_beziers(cc_specs_path, image_save_path, target_specs_path, self.viewpoint_mode, transparent_mode=0)
                    self.image_path_list.append(image_save_path)


            print('-------------------------- End of Iteration ' + str(i) + ' --------------------------')
            print()
            
        self.bezier_list.append(np.vstack((self.p_0, catheter.target_bezier_params_list)))

        catheter.write_reports(self.params_report_path, self.p3d_report_path, self.p2d_report_path)
        
        self.plot_loss(log_scale=True)
        
        file_path = os.path.join(self.images_save_dir, 'bezier_params.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(self.bezier_list, file) 
        fig_path = os.path.join(self.images_save_dir, 'trajectory.png')
        plot_3D_bezier_curve_series(self.bezier_list, fig_path, show=True) 

        return catheter.get_params()
