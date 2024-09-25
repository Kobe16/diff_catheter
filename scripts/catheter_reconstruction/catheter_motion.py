"""
File to rebuild past frame catheters, using inverse and forward kinematics. 
In the end, it should output a reconstructed image of a catheter from k 
frames ago. It will call upon bezier_interspace_transforms for its calculations. 
"""
import sys
import os
import numpy as np

import transforms
import bezier_interspace_transforms
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from bezier_set import BezierSet


class CatheterMotion(): 

    def __init__(self, p_0, l, r, n_mid_points, verbose=1):
        """
        Args:
            p_0 ((3,) numpy array): start point of catheter
            l (float): length of catheter
            r (float): cross section radius of catheter
            n_mid_points (int): number of middle control points
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
        self.bezier_config = np.zeros((6, 1))
