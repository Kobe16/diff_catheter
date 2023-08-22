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
    def __init__(self): 
        '''
        Constructor to initialize the class with set curve & camera parameters
        Also, set manual seed for random number generation --> for reproducibility. 
        '''

        super().__init__()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.epsilon = 1e-8

        # Number of samples to take along Bezier Curve
        self.num_samples = 30
        # Number of samples to take on INSIDE of each circle
        self.samples_per_circle = 1
        # Number of samples to take on OUTSIDE border of each circle
        self.bezier_surface_resolution = 1
        self.bezier_circle_angle_increment = (2 * math.pi) / self.bezier_surface_resolution

        self.cylinder_mesh_points1 = torch.zeros(self.num_samples, self.samples_per_circle, 3)
        self.cylinder_surface_points1 = torch.zeros(self.num_samples, self.bezier_surface_resolution, 3)

        self.cylinder_mesh_points2 = torch.zeros(self.num_samples, self.samples_per_circle, 3)
        self.cylinder_surface_points2 = torch.zeros(self.num_samples, self.bezier_surface_resolution, 3)



        self.radius = 0.0015
        
        # Used to generate same set of random numbers each time (mainly used for getting random points in circles)
        torch.manual_seed(0)

###################################################################################################
###################################################################################################
###################################################################################################

# Helper functions for plotting using matplotlib

    def set_axes_equal(self, ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.

        Args:
            ax (plt.Axes): Matplotlib axes to set equal.
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
        '''
        Set axes radius. Works in conjunction with set_axes_equal. 

        Args:
            ax (plt.Axes): Matplotlib axes to set equal.
            origin (np.array): origin of axes
            radius (float): radius of axes
        '''

        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])

###################################################################################################
###################################################################################################
###################################################################################################

# Helper functions for obtaining the tangent, normal and binormal vectors of a bezier curve

    def getBezierTNB(self, bezier_pos, bezier_der, bezier_snd_der):
        '''
        Get the tangent, normal and binormal vectors of a bezier curve. 
        Add self.epsilon to the denominator to avoid division by zero (to avoid
        getting NaN values in the tensors).

        Args:
            bezier_pos (torch.tensor): bezier curve points
            bezier_der (torch.tensor): bezier curve first derivative
            bezier_snd_der (torch.tensor): bezier curve second derivative
        '''

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
        '''
        Get the normal vector of a bezier curve.
        Add self.epsilon to the denominator to avoid division by zero (to avoid
        getting NaN values in the tensors).

        Args:
            bezier_der (torch.tensor): bezier curve first derivative
            bezier_snd_der (torch.tensor): bezier curve second derivative

        '''

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
        '''
        Get the binormal vector of a bezier curve.
        Add self.epsilon to the denominator to avoid division by zero (to avoid
        getting NaN values in the tensors).

        Args:
            bezier_der (torch.tensor): bezier curve first derivative
            bezier_snd_der (torch.tensor): bezier curve second derivative
        '''

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

        Args:
            set_of_vectors (torch.tensor): set of vectors to normalize

        '''
        normalized_set_of_vectors = set_of_vectors / (torch.linalg.norm(set_of_vectors, ord=2, dim=0) + self.epsilon)
        return normalized_set_of_vectors

    def getTranslatedVectors(self, pos_bezier, set_of_vectors): 
        '''
        Method to get the translated version of a set of vectors (of the shape: (num_samples, 3)). 
        Adds respective point on Bezier curve to the vector (s.t. point is considered 'start' of translated vector). 

        Args:
            pos_bezier (torch.tensor): point on Bezier curve
            set_of_vectors (torch.tensor): set of vectors to translate

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
            radius (int): radius value of circle
            center_point (tensor): center point of circle; i.e., current point on Bezier curve
            normal_vec (tensor): normal vector at that point on Bezier curve
            binormal_vec (tensor): binormal vector at that point on Bezier curve
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
        Method to plot Bezier vectors using MatPlotLib.
        NOTE: Plot will come out weird if using radius that is substantially larger than the 
        length of the actual curve. 


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
        '''
        Method to use in conjunction with plot3dPoints() to plot Bezier curve and TNB vectors.
        '''

        self.ax.set_box_aspect([2,2,2]) 
        self.set_axes_equal(self.ax)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.fig.suptitle('Bézier Curve TNB Frames')

        plt.show()

    def plot3dBezierCylinder(self): 
        '''
        Method to plot 3D cylinder mesh points using MatPlotLib.
        '''

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
        '''
        Method to plot TWO 3D cylinder mesh points using MatPlotLib.

        self.cylinder_mesh_and_surface_points1: prediction model of cylinder
        self.cylinder_mesh_and_surface_points2: ground truth model of cylinder
        '''

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
        Method to obtain FIRST bezier curve position, tangents, normals, and binormals. 
        Calls helper methods to plot these vectors. 

        Args: 
            para_gt: Ground truth parameters for bezier curve. Extract bezier control points from this. 
                     para_gt[0:3] = second control point of quadratic bezier curve
                     para_gt[3:6] = third control point of quadratic bezier curve
            
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
        Method to obtain SECOND bezier curve position, tangents, normals, and binormals. 
        Calls helper methods to plot these vectors. 

        Args: 
            para_gt: Ground truth parameters for bezier curve. Extract bezier control points from this. 
                     para_gt[0:3] = second control point of quadratic bezier curve
                     para_gt[3:6] = third control point of quadratic bezier curve
            
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
    '''
    Main function used to plot the predicted bezier curve and the ground truth bezier curve, 
    together on the same 3D plot. 
    '''

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