'''
File used to run optimization on the catheter parameters. 
Uses Adam optimizer. 
'''
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
from test_loss_define_v2 import ChamferLossWholeImage, ContourChamferLoss, \
    TipChamferLoss, BoundaryPointChamferLoss, TipDistanceLoss, BoundaryPointDistanceLoss, \
    GenerateRefData

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap

from tqdm.notebook import tqdm


class CatheterOptimizeModel(nn.Module): 
    '''
    This class is used to optimize the catheter parameters.
    '''
    def __init__(self, p_start, image_ref, gpu_or_cpu): 
        '''
        This function initializes the catheter optimization model.

        Args:
            p_start (tensor): starting point of the catheter
            image_ref (numpy array): reference image to compare to
            gpu_or_cpu (str): either 'cuda' or 'cpu'
        '''
        super().__init__()

        ###========================================================
        ### 1) SETTING UP BEZIER CURVE CONSTRUCTION
        ###========================================================
        self.build_bezier = ConstructionBezier()
        self.build_bezier.to(gpu_or_cpu)
        self.build_bezier.loadRawImage(img_save_path)

        ###========================================================
        ### 2) SETTING UP LOSS FUNCTIONS
        ###========================================================
        self.generate_ref_data = GenerateRefData()
        self.chamfer_loss_whole_image = ChamferLossWholeImage(device=gpu_or_cpu)
        self.chamfer_loss_whole_image.to(gpu_or_cpu)
        self.contour_chamfer_loss = ContourChamferLoss(device=gpu_or_cpu)
        self.contour_chamfer_loss.to(gpu_or_cpu)
        self.tip_chamfer_loss = TipChamferLoss(device=gpu_or_cpu)
        self.tip_chamfer_loss.to(gpu_or_cpu)
        self.boundary_point_chamfer_loss = BoundaryPointChamferLoss(device=gpu_or_cpu)
        self.boundary_point_chamfer_loss.to(gpu_or_cpu)
        self.tip_distance_loss = TipDistanceLoss(device=gpu_or_cpu)
        self.tip_distance_loss.to(gpu_or_cpu)
        self.boundary_point_distance_loss = BoundaryPointDistanceLoss(device=gpu_or_cpu)
        self.boundary_point_distance_loss.to(gpu_or_cpu)

        # Declare self.tip_euclidean_distance_loss as a variable that'll hold a single numpy scalar value
        self.tip_euclidean_distance_loss = None


        ###========================================================
        ### 3) SETTING UP CURVE PARAMETERS
        ###========================================================

        # Straight Line for initial parameters
        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.02, 0.002, 0.0, 0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)

        self.p_start = p_start.to(gpu_or_cpu).detach()

        self.para_init = nn.Parameter(torch.from_numpy(
            np.array([0.0365, 0.0036,  0.1202,  0.0056, -0.0166, 0.1645],
                     dtype=np.float32)).to(gpu_or_cpu),
                                      requires_grad=True)
        

        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Generate reference data, so you don't need to generate it in every forward pass
        ref_catheter_contour = self.generate_ref_data.get_raw_contour(self.image_ref)
        self.register_buffer('ref_catheter_contour', ref_catheter_contour)
        ref_catheter_centerline = self.generate_ref_data.get_raw_centerline(self.image_ref)
        self.register_buffer('ref_catheter_centerline', ref_catheter_centerline)




    def forward(self, save_img_path): 
        '''
        Function to run forward pass of the catheter optimization model.
        Creates catheter model, gets projection onto 2d image, and calculates loss.

        Args:
            save_img_path (str): path to save the projection image to
        '''

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
        self.build_bezier.getBezierCurveCylinder(self.p_start, self.para_init)

        # cylinder_mesh_points = self.build_bezier.cylinder_mesh_points
        # print("cylinder_mesh_points max value: ", torch.max(cylinder_mesh_points))

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0, 0)
        # loss = torch.mean(torch.norm(cylinder_mesh_points, dim=1))

        # Plot 3D Bezier Cylinder mesh points
        # self.build_bezier.plot3dBezierCylinder()

        # Get 2d projected Bezier Cylinder mesh points
        self.build_bezier.getCylinderMeshProjImg()


        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0)
        # bezier_proj_img = self.build_bezier.bezier_proj_img
        # print("bezier_proj_img:" , bezier_proj_img)
        # print("Max value in bezier_proj_img: ", torch.max(bezier_proj_img))
        # print("average value in bezier_proj_img", torch.mean(bezier_proj_img))
        # loss = torch.mean(torch.norm(bezier_proj_img, dim=1))

        # Get 2d projected Bezier centerline (position) points
        self.build_bezier.getBezierProjImg()

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all points in bezier_proj_centerline_img to (0, 0)
        # bezier_proj_centerline_img = self.build_bezier.bezier_proj_centerline_img
        # loss = torch.mean(torch.norm(bezier_proj_centerline_img, dim=1))

        # Plot 2D projected Bezier Cylinder mesh points
        # print("cylinder_mesh_points: ", self.build_bezier.cylinder_mesh_points)
        self.build_bezier.draw2DCylinderImage(self.image_ref, save_img_path)

        # TODO: add function to save image to file

        ###========================================================
        ### 4) Compute Loss using various Loss Functions
        ###========================================================
        # loss_whole_image = self.chamfer_loss_whole_image(self.build_bezier.bezier_proj_img, self.image_ref)
        loss_contour = self.contour_chamfer_loss(self.build_bezier.bezier_proj_img, self.image_ref)
        loss_tip = self.tip_chamfer_loss(self.build_bezier.bezier_proj_img, self.image_ref)
        loss_boundary = self.boundary_point_chamfer_loss(self.build_bezier.bezier_proj_img, self.image_ref)
        loss_tip_distance, self.tip_euclidean_distance_loss = self.tip_distance_loss(self.build_bezier.bezier_proj_centerline_img, self.image_ref)
        loss_boundary_point_distance_loss = self.boundary_point_distance_loss(self.build_bezier.bezier_proj_img, self.image_ref)

        # loss_whole_image = self.chamfer_loss_whole_image(self.build_bezier.bezier_proj_img, self.image_ref)
        # loss_contour = self.contour_chamfer_loss(self.build_bezier.bezier_proj_img, self.ref_catheter_contour)
        # loss_tip = self.tip_chamfer_loss(self.build_bezier.bezier_proj_img, self.ref_catheter_centerline)
        # loss_boundary = self.boundary_point_chamfer_loss(self.build_bezier.bezier_proj_img, self.ref_catheter_centerline)
        # loss_tip_distance, self.tip_euclidean_distance_loss = self.tip_distance_loss(self.build_bezier.bezier_proj_centerline_img, self.ref_catheter_centerline)
        # loss_boundary_point_distance_loss = self.boundary_point_distance_loss(self.build_bezier.bezier_proj_img, self.ref_catheter_centerline)


        weight = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
        loss = loss_contour * weight[0] + loss_tip * weight[1] + loss_boundary * weight[2] + loss_tip_distance * weight[3] + loss_boundary_point_distance_loss * weight[4]


        print("-----------------------------------------------------------------")
        print("loss_contour: ", loss_contour)
        print("loss_tip: ", loss_tip)
        print("loss_boundary: ", loss_boundary)
        print("loss_tip_distance: ", loss_tip_distance)
        print("loss_boundary_point_distance_loss: ", loss_boundary_point_distance_loss)
        print("loss: ", loss)
        print("-----------------------------------------------------------------")


        # TODO: Plot the loss

        return loss



if __name__ == '__main__':
    '''
    Main function to set up optimzer model and run the optimization loop
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
    # para_init = torch.tensor([0.02, 0.002, 0.0, 0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866], dtype=torch.float)
    # p_start = torch.tensor([0.02, 0.002, 0.0])


    # Z axis + 0.1
    # para_init = torch.tensor([0.02, 0.002, 0.1, 0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866], dtype=torch.float)
    # p_start = torch.tensor([0.02, 0.002, 0.1000])

    # p_start used for SRC presentation
    p_start = torch.tensor([0.02, 0.008, 0.054])

    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/test_diff_render_catheter_v2/blender_imgs/test_catheter_gt1'
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

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

    # Declare loss history lists to keep track of loss values
    proj_end_effector_loss_history = []
    d3d_end_effector_loss_history = []
    # Ground Truth parameters for catheter used in SRC presentation
    para_gt = torch.tensor([0.02003904, 0.0016096, 0.13205799, 0.00489567, -0.03695673, 0.196168896], dtype=torch.float, device=gpu_or_cpu, requires_grad=False)
    end_effector_gt = para_gt[3:6]


    ###========================================================
    ### 3) SET UP AND RUN OPTIMIZATION MODEL
    ###========================================================
    catheter_optimize_model = CatheterOptimizeModel(p_start, img_ref_binary, gpu_or_cpu).to(gpu_or_cpu)

    print("Model Parameters:")
    for name, param in catheter_optimize_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=1e-2)

    # Run the optimization loop
    loop = tqdm(range(100))
    for loop_id in loop:
        print("\n================================================================================================================")
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


        # end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
        proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
        d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())

        # Update the progress bar
        loop.set_description('Optimizing')

        # Update the loss
        loop.set_postfix(loss=loss.item())

        print("Loss: ", loss.item())


    # Given array of values proj_end_effector_loss_history, create plot of loss vs. iterations
    iterations_x_axis_proj = list(range(len(proj_end_effector_loss_history)))
    print("proj_end_effector_loss_history: ", proj_end_effector_loss_history)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig1.suptitle('2D Tip Euclidean Distance Loss History')
    ax1.plot(iterations_x_axis_proj, proj_end_effector_loss_history)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Euclidean Distance Loss (Pixels)')
    ax1.set_xlim([0, len(proj_end_effector_loss_history)])
    ax1.set_ylim([0, 80])
    ax1.grid(True)
    plt.show()


    # Given array of values d3d_end_effector_loss_history, create plot of 3d loss vs. iterations
    iterations_x_axis_3d = list(range(len(d3d_end_effector_loss_history)))
    print("d3d_end_effector_loss_history: ", d3d_end_effector_loss_history)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('3D Tip Euclidean Distance Loss History')
    ax2.plot(iterations_x_axis_3d, d3d_end_effector_loss_history)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Euclidean Distance Loss (m)')
    ax2.set_xlim([0, len(d3d_end_effector_loss_history)])
    ax2.set_ylim([0, 0.05])
    ax2.grid(True)
    plt.show()


