import sys
from turtle import pd

sys.path.append('..')
sys.path.insert(1, '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts')

import os
import numpy as np

# import transforms
# import bezier_interspace_transforms
from bezier_set import BezierSet
import camera_settings

import torch

import open3d as o3d

import cv2
import matplotlib.pyplot as plt

import pdb


class BlenderRenderCatheter:
    '''
    This class works with the BezierSet class to render out a Bezier curve
    in Blender. 
    '''

    def __init__(self):
        '''
        n_beziers: number of Bezier curves to render
        bezier_set: instance of the the BezierSet class
        '''

        ## initialize a catheter
        n_beziers = 1
        self.bezier_set = BezierSet(n_beziers)

    def set_bezier_in_blender(self, para_gt, p_start):
        p_mid = para_gt[0:3]
        p_end = para_gt[3:6]

        # c = (p_mid - (p_start / 4) - (p_end / 4)) * 2
        c1 = 4 / 3 * p_mid - 1 / 3 * p_end
        c2 = 4 / 3 * p_mid - 1 / 3 * p_start

        self.bezier_set.enter_spec(p_start, p_end, c1, c2)

    def render_bezier_in_blender(self,
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


if __name__ == '__main__':

    blender_catheter = BlenderRenderCatheter()

    para_gt = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896], dtype=torch.float)
    p_start = torch.tensor([0.02, 0.002, 0.0])

    # case_naming = '/home/fei/ARCLab-CCCatheter/scripts/diff_render/blender_imgs/diff_render_1'
    # case_naming = '/home/fei/diff_catheter/scripts/diff_render/blender_imgs/diff_render_2'
    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs'
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

    # blender_catheter.set_bezier_in_blender(para_gt.detach().numpy(), p_start.detach().numpy())

    # blender_catheter.render_bezier_in_blender(cc_specs_path, img_save_path, target_specs_path, viewpoint_mode,
    #                                           transparent_mode)

    diff_catheter = ConstructionBezier()

    ## define a bezier curve
    diff_catheter.getBezierCurve(para_gt, p_start)

    ## get the bezier in TNB frame, in order to build a tube mesh
    # diff_catheter.getBezierTNB(diff_catheter.bezier_pos_cam, diff_catheter.bezier_der_cam,
    #                            diff_catheter.bezier_snd_der_cam)
    diff_catheter.getBezierTNB(diff_catheter.bezier_pos, diff_catheter.bezier_der, diff_catheter.bezier_snd_der)

    ## get bezier surface mesh
    ## ref : https://mathworld.wolfram.com/Tube.html
    # diff_catheter.getBezierSurface(diff_catheter.bezier_pos_cam)
    diff_catheter.getBezierSurface(diff_catheter.bezier_pos)

    diff_catheter.createCylinderPrimitive()
    # diff_catheter.createOpen3DVisualizer()
    diff_catheter.updateOpen3DVisualizer()

    # ## load the raw RGB image
    # # diff_catheter.loadRawImage(img_save_path)
    # diff_catheter.proj_bezier_img = diff_catheter.getProjPointCam(diff_catheter.bezier_pos_cam, diff_catheter.cam_K)
    # # diff_catheter.draw2DCenterlineImage()