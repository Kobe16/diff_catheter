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
from test_loss_define_v2 import ChamferLossWholeImage, ContourChamferLoss, TipChamferLoss, BoundaryPointChamferLoss

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap

from tqdm.notebook import tqdm


class CatheterOptimizeModel(nn.Module): 
    def __init__(self, image_ref, gpu_or_cpu): 
        super().__init__()

        ###========================================================
        ### 1) SETTING UP BEZIER CURVE CONSTRUCTION
        ###========================================================
        self.build_bezier = ConstructionBezier()
        self.build_bezier.to(gpu_or_cpu)
        self.build_bezier.loadRawImage(img_save_path)

        self.chamfer_loss_whole_image = ChamferLossWholeImage(device=gpu_or_cpu)
        self.chamfer_loss_whole_image.to(gpu_or_cpu)
        self.contour_chamfer_loss = ContourChamferLoss(device=gpu_or_cpu)
        self.contour_chamfer_loss.to(gpu_or_cpu)
        self.tip_chamfer_loss = TipChamferLoss(device=gpu_or_cpu)
        self.tip_chamfer_loss.to(gpu_or_cpu)
        self.boundary_point_chamfer_loss = BoundaryPointChamferLoss(device=gpu_or_cpu)
        self.boundary_point_chamfer_loss.to(gpu_or_cpu)

        # Straight Line for initial parameters
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.02, 0.002, 0.0, 0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)

        # Z axis + 0.1
        self.para_init = nn.Parameter(torch.from_numpy(
            np.array([0.02, 0.002, 0.1, 0.0096, -0.0080,  0.1969, -0.0414, -0.0131,  0.2820],
                     dtype=np.float32)).to(gpu_or_cpu),
                                      requires_grad=True)
        

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
        self.build_bezier.getBezierCurveCylinder(self.para_init)

        # cylinder_mesh_points = self.build_bezier.cylinder_mesh_points
        # print("cylinder_mesh_points max value: ", torch.max(cylinder_mesh_points))

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0, 0)
        # loss = torch.mean(torch.norm(cylinder_mesh_points, dim=1))

        # Plot 3D Bezier Cylinder mesh points
        # self.build_bezier.plot3dBezierCylinder()

        # Get 2d projected Bezier Cylinder mesh points
        self.build_bezier.getCylinderMeshProjImg()

        # Plot 2D projected Bezier Cylinder mesh points
        # print("cylinder_mesh_points: ", self.build_bezier.cylinder_mesh_points)
        self.build_bezier.draw2DCylinderImage(self.image_ref, save_img_path)

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0)
        # bezier_proj_img = self.build_bezier.bezier_proj_img
        # print("bezier_proj_img:" , bezier_proj_img)
        # print("Max value in bezier_proj_img: ", torch.max(bezier_proj_img))
        # print("average value in bezier_proj_img", torch.mean(bezier_proj_img))
        # loss = torch.mean(torch.norm(bezier_proj_img, dim=1))


        # TODO: add function to save image to file

        ###========================================================
        ### 4) Compute Loss using various Loss Functions
        ###========================================================
        # loss = self.chamfer_loss_whole_image(self.build_bezier.bezier_proj_img, self.image_ref)
        loss_contour = self.contour_chamfer_loss(self.build_bezier.bezier_proj_img, self.image_ref)
        loss_tip = self.tip_chamfer_loss(self.build_bezier.bezier_proj_img, self.image_ref)
        loss_boundary = self.boundary_point_chamfer_loss(self.build_bezier.bezier_proj_img, self.image_ref)

        weight = torch.tensor([1.0, 1.0, 1.0])
        loss = loss_contour * weight[0] + loss_tip * weight[1] + loss_boundary * weight[2]

        print("-----------------------------------------------------------------")
        print("loss_contour: ", loss_contour)
        print("loss_tip: ", loss_tip)
        print("loss_boundary: ", loss_boundary)
        print("loss: ", loss)
        print("-----------------------------------------------------------------")


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
    # para_init = torch.tensor([0.02, 0.002, 0.0, 0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866], dtype=torch.float)
    # p_start = torch.tensor([0.02, 0.002, 0.0])

    # Z axis + 0.1
    # para_init = torch.tensor([0.02, 0.002, 0.1, 0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866], dtype=torch.float)
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
    ### 3) SET UP AND RUN OPTIMIZATION MODEL
    ###========================================================
    catheter_optimize_model = CatheterOptimizeModel(img_ref_binary, gpu_or_cpu).to(gpu_or_cpu)


    print("Model Parameters:")
    for name, param in catheter_optimize_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=1e-2)

    # Run the optimization loop
    loop = tqdm(range(100))
    for loop_id in loop:
        print("\n========================================================")
        print("loop_id: ", loop_id)


        save_img_path = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/test_diff_render_catheter_v2/rendered_imgs/' \
            + 'render_' + str(loop_id) + '.jpg'

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


