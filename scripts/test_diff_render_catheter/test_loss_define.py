import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

import skimage.morphology as skimage_morphology
import cv2


class AppearanceLoss(nn.Module): 

    def __init__(self, device): 
        super(AppearanceLoss, self).__init__()
        self.device = device

    def forward(self, img_render_proj_pts, img_ref): 
        '''Method to compute Appearance Loss between image of projected points and 
            original reference image. Will calculate the loss by forcing all proj pixels
            to be white. 

        Args: 
            img_render_proj_pts: Image of projected points. 
            img_ref: Original reference image.
        '''

        img_render_proj_pts = 1 / (1 + torch.exp(-100 * (img_render_proj_pts - 0.1)))  # Differentiable binarization (approximation)
        mask = (img_render_proj_pts > 0.1)
        img_render_proj_pts = img_render_proj_pts * mask  # Zero out values above the threshold 0.5

        print("img_render_proj_pts.shape: ", img_render_proj_pts.shape)
        print("img_render_proj_pts: ", img_render_proj_pts)
        
        print("img_ref.shape: ", img_ref.shape)
        print("img_ref: " , img_ref)



        # # Grayscale the reference image
        # img_ref_gray = cv2.cvtColor(img_render_proj_pts, cv2.COLOR_BGR2GRAY)

        # # Applies binary thresholding operation to grayscale img. 
        # # Sets all pixel values below the threshold value of 80 to 0. 
        # # Sets all pixel values above or equal to 80 to 255
        # (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # # Creates a binary image by replacing all pixel values equal to 255 with 1 (leaves other pixel values unchanged) 
        # img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)


if __name__ == '__main__': 
    a = MaskLoss(torch.device("cpu"))
    a.forward(1, 2)