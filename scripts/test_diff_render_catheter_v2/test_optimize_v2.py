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

from test_reconst_v2 import ConstructionBezier
# from blender_catheter import BlenderRenderCatheter
# from diff_render_catheter import DiffRenderCatheter
from test_loss_define_v2 import AppearanceLoss, CenterlineLoss, ChamferLossWholeImage, ContourChamferLoss

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
        self.centerline_loss = CenterlineLoss(device=gpu_or_cpu)
        self.centerline_loss.to(gpu_or_cpu)
        self.chamfer_loss_whole_image = ChamferLossWholeImage(device=gpu_or_cpu)
        self.chamfer_loss_whole_image.to(gpu_or_cpu)
        self.contour_chamfer_loss = ContourChamferLoss(device=gpu_or_cpu)
        self.contour_chamfer_loss.to(gpu_or_cpu)

        self.p_start = p_start.to(gpu_or_cpu).detach()

        # Straight Line for initial parameters
        self.para_init = nn.Parameter(torch.from_numpy(
            np.array([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
                     dtype=np.float32)).to(gpu_or_cpu),
                                      requires_grad=True)
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.02958988, 0.01195899, 0.10690406, -0.02142905, 0.0068571, 0.19200866],
        #         dtype=np.float32)).to(gpu_or_cpu),
        #                         requires_grad=True)
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.03958988, 0.02195899, 0.11690406, 0.11690406, 0.0168571, 0.20200866],
        #         dtype=np.float32)).to(gpu_or_cpu),
        #                         requires_grad=True)
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.08958988, 0.07195899, 0.16690406, 0.16690406, 0.0668571, 0.25200866],
        #         dtype=np.float32)).to(gpu_or_cpu),
        #                         requires_grad=True)

        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([ 0.0096, -0.0080,  0.0869, -0.0414, -0.0131,  0.1720],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)

        # Z axis + 0.05
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([ 0.0096, -0.0080,  0.1469, -0.0414, -0.0131,  0.2320],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)

        # Z axis + 0.1
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([ 0.0096, -0.0080,  0.1969, -0.0414, -0.0131,  0.2820],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)
        

        # Get the silhouette of the reference RGB image by finding all non-white pixel values.
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)

    def forward(self, save_img_path): 
        # # Get 2d center line from reference image (using skeletonization)
        # centerline_ref = self.centerline_loss.get_raw_centerline(self.image_ref)
        # print("centerline_ref shape: ", centerline_ref.shape)
        # print("centerline_ref: ", centerline_ref)
        
        # # Plot the points in centerline_ref 
        # fig1, ax1 = plt.subplots()
        # ax1.plot(centerline_ref[:, 1], centerline_ref[:, 0])
        # ax1.set_title('centerline_ref')
        # ax1.set_xlim([0, 640])
        # ax1.set_ylim([480, 0])
        # plt.show()


        ###========================================================
        ### 1) RUNNING BEZIER CURVE CONSTRUCTION
        ###========================================================
        # Generate the Bezier curve cylinder mesh points
        self.build_bezier.getBezierCurveCylinder(self.para_init, self.p_start)

        # cylinder_mesh_points = self.build_bezier.cylinder_mesh_points
        # print("cylinder_mesh_points max value: ", torch.max(cylinder_mesh_points))

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0, 0)
        # loss = torch.mean(torch.norm(cylinder_mesh_points, dim=1))

        # Plot 3D Bezier Cylinder mesh points
        # self.build_bezier.plot3dBezierCylinder()

        # Plot 2D projected Bezier Cylinder mesh points
        self.build_bezier.getCylinderMeshProjImg()
        # print("cylinder_mesh_points: ", self.build_bezier.cylinder_mesh_points)
        # self.build_bezier.draw2DCylinderImage()

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0)
        # bezier_proj_img = self.build_bezier.bezier_proj_img
        # print("bezier_proj_img:" , bezier_proj_img)
        # print("Max value in bezier_proj_img: ", torch.max(bezier_proj_img))
        # print("average value in bezier_proj_img", torch.mean(bezier_proj_img))
        # loss = torch.mean(torch.norm(bezier_proj_img, dim=1))

        ###========================================================
        ### 2) Get 2D projected points image from bezier curve tube
        ###========================================================
        # img_render = self.build_bezier.get2DCylinderImage()
        # TODO: add function to save image to file


        ###========================================================
        ### 3) Compute Appearance loss between projected points image and reference image
        ###========================================================
        # Extract alpha channel from img_render, then convert to torch tensor
        # img_render_alpha = torch.from_numpy(img_render[0, ..., 3].astype(np.float32))
        # print("img_render_alpha: ", img_render_alpha.shape)
        # fig, ax = plt.subplots()
        # ax.plot(img_render_alpha)
        # ax.set_title('img_render_alpha')
        # plt.imshow(img_render_alpha)
        # plt.show()

        ###========================================================
        ### 4) Compute Chamfer Distance loss between projected points image and reference image points
        ###========================================================
        # loss = self.chamfer_loss_whole_image(self.build_bezier.bezier_proj_img, self.image_ref)
        loss = self.contour_chamfer_loss(self.build_bezier.bezier_proj_img, self.image_ref)

        # Compute loss between rendered image and reference image
        # loss, img_render_binary = self.appearance_loss(img_render_alpha.unsqueeze(0), self.image_ref.unsqueeze(0))

        # Make loss require grad
        # loss.requires_grad = True

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
    # p_start = torch.tensor([0.03, 0.012, 0.01])
    # p_start = torch.tensor([0.04, 0.022, 0.02])
    # p_start = torch.tensor([0.09, 0.072, 0.07])

    # p_start = torch.tensor([ 0.0100, -0.0080, -0.0100])

    # Z axis + 0.05
    # p_start = torch.tensor([0.02, 0.002, 0.0500])

    # Z axis + 0.1
    # p_start = torch.tensor([0.02, 0.002, 0.1000])




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

    optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=1e-2)

    # Run the optimization loop
    loop = tqdm(range(200))
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
        loss.backward(retain_graph=True)

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
            print(f"Parameter: {name}, Updated Value: {param.data}")


        # Update the progress bar
        loop.set_description('Optimizing')

        # Update the loss
        loop.set_postfix(loss=loss.item())

        print("Loss: ", loss.item())


