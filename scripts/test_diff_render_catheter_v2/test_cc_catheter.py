import random
import numpy as np
import cv2

import test_transforms
import test_bezier_interspace_transforms
# from bezier_set import BezierSet

class CCCatheter(): 
    def __init__(self, p0, r, ): 
        """
        Initialize a Catheter Model object.

        Args:
            p_0 ((3,) numpy array): start point of catheter
            l (float): length of catheter
            r (float): cross section radius of catheter

        Attributes:
            p0 (float): Initial parameter p0.
            r (float): Parameter r.
            mode (int or None): Indicates the number of Degrees of Freedom (DoF).
            weight_matrix (numpy.ndarray): Matrix for weight calculations.
            params (numpy.ndarray): Array to store catheter parameters over iterations.
        """
        
        self.p0 = p0
        self.r = r

        self.mode = None
        self.weight_matrix = np.zeros((3, 3))
        self.params = np.zeros((self.n_iter + 2, 5))

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


    def get_bezier_reconstruction(self): 
        """
        Get the bezier params of a catheter from a picture of the catheter. 
        Call's Kobe's summer 2023 catheter reconstruction script
        """

    def update_2dof_bezier_interspace_ux_uy(self, bezier_reconst, bezier_t0, ux_t0, damping_const): 
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

        # TODO: replace this inverse kinematics math with the damped least squares method
        # TODO: change name of this method. It shouldn't be called update.... because its
        #       its goal is not to update parameters to simulate movement. Its goal is to 
        #       just convert between the difference interspaces. I think I originally named
        #       it 'update' because I just copied the code over from cc_catheter.py

        # Calculate jacobian for current state of catheter
        J = test_bezier_interspace_transforms.calculate_jacobian_2dof_ux_uy(self.p0, self.ux, self.uy, self.l, self.r)

        # Get Jacobian inverse J_inv
        J_inv = np.linalg.inv(J)

        u_delta = damping_const * J_inv * (bezier_t0 - bezier_reconst)

        ux = ux_t0 - u_delta

        return ux
    
    def 


