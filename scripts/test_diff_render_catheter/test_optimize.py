import sys
from turtle import pd

sys.path.append('..')

import os
import numpy as np

# import transforms
# import bezier_interspace_transforms
sys.path.insert(1, '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts')
from bezier_set import BezierSet
import camera_settings

import torch

import open3d as o3d

import cv2
import matplotlib.pyplot as plt

import pdb

from test_reconst import ConstructionBezier
# from blender_catheter import BlenderRenderCatheter
# from diff_render_catheter import DiffRenderCatheter
from test_loss_define import AppearanceLoss

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap

from tqdm.notebook import tqdm


class CatheterOptimizeModel(nn.Module): 
    def __init__(self, p_start, image_ref, gpu_or_cpu): 
        super().__init__()

        self.build_bezier = ConstructionBezier()
        self.build_bezier.to(gpu_or_cpu)

        self.mask_loss = AppearanceLoss(device=gpu_or_cpu)
        self.mask_loss.to(gpu_or_cpu)

        self.p_start = p_start.to(gpu_or_cpu)

        # Straight Line for initial parameters
        self.para_init = nn.Parameter(torch.from_numpy(
            np.array([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
                     dtype=np.float32)).to(gpu_or_cpu),
                                      requires_grad=True)

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)

    def forward(self, save_img_path): 
        print("todo")

        ###========================================================
        ### Get 3D projected points from bezier curve
        ###========================================================
        self.build_bezier.getBezierCurveCylinder()

if __name__ == '__main__':

    p_start = torch.tensor([0.02, 0.002, 0.0])
    para_init = torch.tensor([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
                             dtype=torch.float)
    
    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/diff_render_1'
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

    img_id = 0
    save_img_path = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/' + 'torch3d_render_' + str(
        img_id) + '.jpg'  # save the figure to file
    
    # Set to GPU or CPU computing
    if torch.cuda.is_available():
        gpu_or_cpu = torch.device("cuda:0")
        torch.cuda.set_device(gpu_or_cpu)
    else:
        gpu_or_cpu = torch.device("cpu")


    # Obtain reference image. Create binary mask of catheter
    img_ref_rgb = cv2.imread(img_save_path)

    # Grayscale the reference image
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)

    # Applies binary thresholding operation to grayscale img. 
    # Sets all pixel values below the threshold value of 80 to 0. 
    # Sets all pixel values above or equal to 80 to 255
    (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Creates a binary image by replacing all pixel values equal to 255 with 1 (leaves other pixel values unchanged) 
    img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)
    



