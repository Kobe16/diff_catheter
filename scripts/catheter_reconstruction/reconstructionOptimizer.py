"""
New version, complete
"""

import sys
sys.path.append('..')
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import torch
import torch.nn as nn

import numpy as np

from construction_bezier import ConstructionBezier
from loss_define import (
    ContourChamferLoss, 
    TipDistanceLoss, 
    ImageContourChamferLoss, 
    GenerateRefData
)

from catheter_motion_tensor import CatheterMotion
from utils import *

class ReconstructionOptimizer(nn.Module): 

    def __init__(self, p_start, para_init, image_ref, gpu_or_cpu, past_frames_list, delta_u_list, img_save_path, image_save_path_list): 
        '''
        This function initializes the catheter optimization model.

        Args:
            p_start (tensor, length = 3): starting point of the catheter
            para_init (tensor, length = 6): initial guess for the catheter parameters
            image_ref (numpy array): ground truth image
            gpu_or_cpu (str): either 'cuda' or 'cpu'
            past_frames_list (list of numpy arrays): list of past frames
            delta_u_list (tensor): list of delta u values, torch.tensor([[ux_1, uy_1], [ux_1, uy_1], ...])
            img_save_path (str): path of the ground truth image
            image_save_path_list (list of str): list of paths of the past frames
        '''
        super().__init__()

        """
        NOTE: To initialize ConstructionBezier class as an instance object as ReconstructionOptimizer 
        during its initialization, it will cause the issue that gradient computation complexity 
        accumulates with the number of iterations. Because during the forward pass of each iteration, 
        some intermediate variables may be saved as instance variables of the ConstructionBezier class 
        and thus referenced in the backward pass of the next iteration, 
        resulting in each iteration not being independent of each other.
        """
        # self.build_bezier = ConstructionBezier(radius=0.0015)
        # self.build_bezier.to(gpu_or_cpu)
        # self.build_bezier.loadRawImage(img_save_path)
        
        self.img_save_path = img_save_path

        self.contour_chamfer_loss = ContourChamferLoss(device=gpu_or_cpu)
        self.contour_chamfer_loss.to(gpu_or_cpu)
        self.tip_distance_loss = TipDistanceLoss(device=gpu_or_cpu)
        self.tip_distance_loss.to(gpu_or_cpu)
        self.image_contour_chamfer_loss = ImageContourChamferLoss(device=gpu_or_cpu)
        self.image_contour_chamfer_loss.to(gpu_or_cpu)
        
        # Declare self.tip_euclidean_distance_loss as a variable that'll hold a single numpy scalar value
        self.tip_euclidean_distance_loss = None
        self.tip_loss = None

        self.p_start = p_start.to(gpu_or_cpu).detach()
        self.para_init = nn.Parameter(torch.from_numpy(para_init).to(gpu_or_cpu),
                                      requires_grad=True)
        
        
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Generate reference data, so you don't need to generate it in every forward pass
        self.generate_ref_data = GenerateRefData(self.image_ref)
        ref_catheter_contour = self.generate_ref_data.get_raw_contour()
        # self.register_buffer('ref_catheter_contour', ref_catheter_contour)
        self.ref_catheter_contour = ref_catheter_contour
        ref_catheter_centerline = self.generate_ref_data.get_raw_centerline()
        # self.register_buffer('ref_catheter_centerline', ref_catheter_centerline)
        self.ref_catheter_centerline = ref_catheter_centerline
        
        # self.register_buffer('delta_u_list', delta_u_list)
        self.delta_u_list = delta_u_list.to(gpu_or_cpu)
        # self.register_buffer('past_frames_list', past_frames_list)
        # self.register_buffer('image_save_path_list', image_save_path_list)
        self.image_save_path_list = image_save_path_list
        
        # Generate reference data for past frames
        contour_list = []
        centerline_list = []
        for image in past_frames_list:
            image = torch.from_numpy(image.astype(np.float32))
            generate_ref_data = GenerateRefData(image)
            ref_catheter_contour = generate_ref_data.get_raw_contour()
            ref_catheter_centerline = generate_ref_data.get_raw_centerline()
            contour_list.append(ref_catheter_contour)
            centerline_list.append(ref_catheter_centerline)
        # self.register_buffer('contour_list', torch.stack(contour_list))
        # self.register_buffer('centerline_list', torch.stack(centerline_list))
        self.contour_list = contour_list
        self.centerline_list = centerline_list
        
        self.gpu_or_cpu = gpu_or_cpu

    def forward(self, save_img_path=None): 
        '''
        Function to run forward pass of the catheter optimization model.
        Creates catheter model, gets projection onto 2d image, and calculates loss.

        Args:
            save_img_path (str): path to save the projection image to
        '''
        
        build_bezier = ConstructionBezier(radius=0.0015)
        build_bezier.to(self.gpu_or_cpu)
        build_bezier.loadRawImage(self.img_save_path)
        
        # Generate the Bezier curve cylinder mesh points
        build_bezier.getBezierCurveCylinder(self.p_start, self.para_init)

        # Get 2d projected Bezier Cylinder mesh points
        build_bezier.getCylinderMeshProjImg()

        # Get 2d projected Bezier centerline (position) points
        build_bezier.getBezierProjImg()

        loss_contour = self.contour_chamfer_loss(build_bezier.bezier_proj_img.to(self.gpu_or_cpu), self.ref_catheter_contour.to(self.gpu_or_cpu))

        loss_tip_distance, self.tip_euclidean_distance_loss = self.tip_distance_loss(build_bezier.bezier_proj_centerline_img.to(self.gpu_or_cpu), self.ref_catheter_centerline.to(self.gpu_or_cpu))
        
        catheterMotion = CatheterMotion(self.p_start, self.gpu_or_cpu, l=0.2, r=0.01)
        predicted_paras = catheterMotion.past_frames_prediction(self.delta_u_list, self.para_init)
        # print("predicted_paras.requires_grad:", predicted_paras.requires_grad)
        motion_model_loss = torch.tensor(0.0).to(self.gpu_or_cpu)
        for i in range(len(predicted_paras)):
            construction_bezier = ConstructionBezier()
            construction_bezier.to(self.gpu_or_cpu)
            construction_bezier.loadRawImage(self.image_save_path_list[i])
            construction_bezier.getBezierCurveCylinder(self.p_start, predicted_paras[i].to(self.gpu_or_cpu))
            construction_bezier.getCylinderMeshProjImg()
            construction_bezier.getBezierProjImg()
            
            loss_contour_m = self.contour_chamfer_loss(construction_bezier.bezier_proj_img.to(self.gpu_or_cpu), self.contour_list[i].to(self.gpu_or_cpu))
            motion_model_loss += loss_contour_m.to(self.gpu_or_cpu)
            
            loss_tip_distance_m, self.tip_loss = self.tip_distance_loss(construction_bezier.bezier_proj_centerline_img.to(self.gpu_or_cpu), self.centerline_list[i].to(self.gpu_or_cpu)) 
            # weight_m = torch.tensor([1.0, 1.0]).to(self.gpu_or_cpu)
            # loss_m = loss_contour_m.to(self.gpu_or_cpu) * weight_m[0] + loss_tip_distance_m.to(self.gpu_or_cpu) * weight_m[1]
            # motion_model_loss += loss_m.to(self.gpu_or_cpu)
            motion_model_loss += loss_tip_distance_m.to(self.gpu_or_cpu)
            
        
        weight = torch.tensor([1.0, 1.0, 1.0]).to(self.gpu_or_cpu)
        # weight = torch.tensor([1.0e-5, 1.0, 1.0e-6]).to(self.gpu_or_cpu)
        loss = loss_contour.to(self.gpu_or_cpu) * weight[0] + loss_tip_distance.to(self.gpu_or_cpu) * weight[1] + motion_model_loss * weight[2] / len(predicted_paras)
        # loss = loss_tip_distance.to(self.gpu_or_cpu) * weight[1] + motion_model_loss * weight[2]
        
        if save_img_path is not None:
            build_bezier.draw2DCylinderImage(self.image_ref, save_img_path)


        # print("-----------------------------------------------------------------")
        # print("loss_contour: ", loss_contour)
        # # print("loss_tip: ", loss_tip)
        # # print("loss_boundary: ", loss_boundary)
        # print("loss_tip_distance: ", loss_tip_distance)
        # # print("loss_boundary_point_distance_loss: ", loss_boundary_point_distance_loss)
        # print("motion_model_loss: ", motion_model_loss)
        # print("loss: ", loss)
        # print("-----------------------------------------------------------------")


        # TODO: Plot the loss
        self.loss = loss
        self.loss_contour = loss_contour
        self.loss_tip_distance = loss_tip_distance
        self.motion_model_loss = motion_model_loss

        return loss