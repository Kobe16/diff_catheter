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

        ###========================================================
        ### 1) SETTING UP BEZIER CURVE CONSTRUCTION
        ###========================================================
        self.build_bezier = ConstructionBezier()
        self.build_bezier.to(gpu_or_cpu)
        self.build_bezier.loadRawImage(img_save_path)

        self.appearance_loss = AppearanceLoss(device=gpu_or_cpu)
        self.appearance_loss.to(gpu_or_cpu)

        self.p_start = p_start.to(gpu_or_cpu)

        self.bezier_radius = 0.0015

        # Straight Line for initial parameters
        self.para_init = nn.Parameter(torch.from_numpy(
            np.array([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
                     dtype=np.float32)).to(gpu_or_cpu),
                                      requires_grad=True)

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)

    def forward(self, save_img_path): 
        ###========================================================
        ### 1) RUNNING BEZIER CURVE CONSTRUCTION
        ###========================================================
        # Generate the Bezier curve cylinder mesh points
        self.build_bezier.getBezierCurveCylinder(self.para_init, self.p_start, self.bezier_radius)

        # Plot 3D Bezier Cylinder mesh points
        # self.build_bezier.plot3dBezierCylinder()

        # Plot 2D projected Bezier Cylinder mesh points
        self.build_bezier.getCylinderMeshProjImg()
        # self.build_bezier.draw2DCylinderImage()

        ###========================================================
        ### 2) Get 2D projected points image from bezier curve tube
        ###========================================================
        img_render = self.build_bezier.get2DCylinderImage()
        # TODO: add function to save image to file

        ###========================================================
        ### 3) Compute Appearance loss between projected points image and reference image
        ###========================================================
        # Extract alpha channel from img_render, then convert to torch tensor
        img_render_alpha = torch.from_numpy(img_render[0, ..., 3].astype(np.float32))
        # print("img_render_alpha: ", img_render_alpha.shape)
        # fig, ax = plt.subplots()
        # ax.plot(img_render_alpha)
        # ax.set_title('img_render_alpha')
        # plt.imshow(img_render_alpha)
        # plt.show()
        

        # Compute loss between rendered image and reference image
        loss, img_render_binary = self.appearance_loss(img_render_alpha.unsqueeze(0), self.image_ref.unsqueeze(0))
        # print("loss: ", loss)

        # Make loss require grad
        loss.requires_grad = True

        # TODO: Plot the loss

        return loss



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
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

    img_id = 0
    save_img_path = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/' + 'torch3d_render_' + str(
        img_id) + '.jpg'  # save the figure to file

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
    ### 3) SET UP OPTIMIZATION MODEL
    ###========================================================
    catheter_optimize_model = CatheterOptimizeModel(p_start, img_ref_binary, gpu_or_cpu).to(gpu_or_cpu)


    print("Model Parameters:")
    for name, param in catheter_optimize_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=1e-4)

    # Run the optimization loop
    loop = tqdm(range(10))
    for loop_id in loop:
        print("\n========================================================")
        print("loop_id: ", loop_id)

        # pdb.set_trace()

        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()

        # Run the forward pass
        loss = catheter_optimize_model(save_img_path)

        # Print gradients for all parameters before backward pass
        print("Gradients BEFORE BACKWARD PASS:")
        for name, param in catheter_optimize_model.named_parameters():
            if param.grad is not None:
                print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")  # Print the norm of the gradient
            else:
                print(f"{name}: No gradient computed")

        # Run the backward pass
        loss.backward()

        # Print gradients for all parameters after backward pass
        print("Gradients AFTER BACKWARD PASS:")
        for name, param in catheter_optimize_model.named_parameters():
            if param.grad is not None:
                print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")
            else:
                print(f"{name}: No gradient computed")

        # Update the parameters
        optimizer.step()

        # Print and inspect the updated parameters
        for name, param in catheter_optimize_model.named_parameters():
            print(f"Parameter: {name}, Updated Value: {param.data.norm().item()}")


        # Update the progress bar
        loop.set_description('Optimizing')

        # Update the loss
        loop.set_postfix(loss=loss.item())

        print("Loss: ", loss.item())


