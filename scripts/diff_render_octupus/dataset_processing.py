import torch

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap
import matplotlib.pyplot as plt

import skimage.morphology as skimage_morphology
import cv2

import pdb

import numpy as np
import scipy


class DatasetProcess():

    def __init__(self, centerline_gt, radius_gt, frame_id):
        self.centerline_gt = centerline_gt.copy()
        self.radius_gt = radius_gt.copy()
        self.frame_id = frame_id

    def get_initial_guess_cubic_bezier(self):

        tangent_start = self.centerline_gt[self.frame_id, :, 1] - self.centerline_gt[self.frame_id, :, 0]
        tangent_end = self.centerline_gt[self.frame_id, :, -2] - self.centerline_gt[self.frame_id, :, -1]

        # using the middle point as the curve length calculation
        # pt_middle = self.centerline_gt[self.frame_id, :, int(self.centerline_gt.shape[2] / 2)]
        # scale = np.linalg.norm(pt_middle - self.centerline_gt[self.frame_id, :, 0], ord=None, axis=0, keepdims=False)

        scale_start = 10
        scale_end = 0.1

        pt0 = self.centerline_gt[self.frame_id, :, 0]
        pt1 = pt0 + tangent_start / np.linalg.norm(tangent_start, ord=None, axis=0, keepdims=False) * scale_start
        pt3 = self.centerline_gt[self.frame_id, :, -1]
        pt2 = pt3 + tangent_end / np.linalg.norm(tangent_end, ord=None, axis=0, keepdims=False) * scale_end

        return (pt0, pt1, pt2, pt3)

    def get_initial_guess_quadratic_bezier(self, frame_id):

        pt_start = self.centerline_gt[frame_id, :, 0]
        pt_start_tang = self.centerline_gt[frame_id, :, 10]

        pt_end = self.centerline_gt[frame_id, :, -1]
        pt_end_tang = self.centerline_gt[frame_id, :, -2]

        pt_mid = self.get_intersect(pt_start, pt_start_tang, pt_end, pt_end_tang)

        return (
            pt_start,
            pt_mid,
            pt_end,
        )

    def get_initial_guess_quadratic_bezier_frame0(self):

        pt_start = self.centerline_gt[0, :, 0]

        pt_end = self.centerline_gt[0, :, -1]

        pt_mid = (pt_start + pt_end) / 2

        return (
            pt_start,
            pt_mid,
            pt_end,
        )

    # takes in two lines, the line formed by pt1 and pt2, and the line formed by pt3 and pt4, and finds their intersection or closest point
    # please ref : https://stackoverflow.com/questions/44631259/line-line-intersection-in-python-with-numpy
    # another ref (with analytical form) : https://stackoverflow.com/questions/2316490/the-algorithm-to-find-the-point-of-intersection-of-two-3d-line-segment
    def get_intersect(self, pt1, pt2, pt3, pt4):
        #least squares method
        def errFunc(estimates):
            s, t = estimates
            x = pt1 + s * (pt2 - pt1) - (pt3 + t * (pt4 - pt3))
            return x

        estimates = [1, 1]

        sols = scipy.optimize.least_squares(errFunc, estimates)
        s, t = sols.x

        x1 = pt1[0] + s * (pt2[0] - pt1[0])
        x2 = pt3[0] + t * (pt4[0] - pt3[0])
        y1 = pt1[1] + s * (pt2[1] - pt1[1])
        y2 = pt3[1] + t * (pt4[1] - pt3[1])
        z1 = pt1[2] + s * (pt2[2] - pt1[2])
        z2 = pt3[2] + t * (pt4[2] - pt3[2])

        x = (x1 + x2) / 2  # halfway point if they don't match
        y = (y1 + y2) / 2  # halfway point if they don't match
        z = (z1 + z2) / 2  # halfway point if they don't match

        pt_intersect = np.asarray([x, y, z])

        return pt_intersect