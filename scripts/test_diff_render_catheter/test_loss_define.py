from matplotlib import pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

import skimage.morphology as skimage_morphology
import cv2

from test_reconst import ConstructionBezier


class AppearanceLoss(nn.Module): 

    def __init__(self, device): 
        super(AppearanceLoss, self).__init__()
        self.device = device
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, img_render, img_ref): 
        '''Method to compute Appearance Loss between image of projected points and 
            original reference image. Will calculate the loss by forcing all proj pixels
            to be white. 

        Args: 
            img_render: Image of projected points (not binarized yet). 
            img_ref: Original reference image, has been binarized (turned to sillouhette).
        '''


        # ACTUAL CODE
        img_render_binary = img_render.squeeze()
        img_ref = img_ref.squeeze()

        # print("img_render_binary.shape: ", img_render_binary.shape)
        # print("img_render_binary: ", img_render_binary)
        
        # print("img_ref.shape: ", img_ref.shape)
        # print("img_ref: " , img_ref)

        # Plot img_ref and img_render_binary
        # ISSUE: blank screen (no proj points on img_render_binary)
        # plt.figure()
        # plt.imshow(img_ref)
        # plt.show()

        # plt.figure()
        # plt.imshow(img_render_binary)
        # plt.show()


        dist = torch.sum((img_render - img_ref) ** 2)
        # dist = self.mse_loss(img_render, img_ref)
        assert (dist >= 0)

        return dist, img_render_binary





if __name__ == '__main__': 

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
    para_init = torch.tensor([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866], dtype=torch.float)
    p_start = torch.tensor([0.02, 0.002, 0.0])

    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/diff_render_1'
    img_save_path = case_naming + '.png'

    '''
    Create binary mask of catheter: 
        1) Grayscale the ref img, 
        2) threshold the grayscaled img, 
        3) Creates a binary image by replacing all 
           pixel values equal to 255 with 1 (leaves
           other pixel values unchanged)
    '''
    img_ref_rgb = cv2.imread(img_save_path)
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)
    (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)


    ###========================================================
    ### 3) SETTING UP BEZIER CURVE CONSTRUCTION
    ###========================================================
    build_bezier = ConstructionBezier()
    build_bezier.loadRawImage(img_save_path)


    ###========================================================
    ### 4) RUNNING BEZIER CURVE CONSTRUCTION
    ###========================================================
    # Generate the Bezier curve cylinder mesh points
    build_bezier.getBezierCurveCylinder(para_init, p_start, 0.01 * 0.1)

    # Plot 3D Bezier Cylinder mesh points
    build_bezier.plot3dBezierCylinder()

    # Plot 2D projected Bezier Cylinder mesh points
    build_bezier.getCylinderMeshProjImg()
    build_bezier.draw2DCylinderImage()


    ###========================================================
    ### 4) GET BINARIZED IMAGE OF PROJECTED POINTS
    ###========================================================

    img_proj_pts_bin = build_bezier.get2DCylinderImage()

    # print("img_proj_pts_bin: " + str(img_proj_pts_bin))

    ###========================================================
    ### 4) GET APPEARANCE LOSS
    ###========================================================


    a_loss = AppearanceLoss(torch.device("cpu"))
    loss, img_render_binary = a_loss.forward(img_proj_pts_bin, img_ref_binary)
    print("loss: ", loss)
