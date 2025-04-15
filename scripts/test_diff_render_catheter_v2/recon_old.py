"""
Old version, complete
"""

import sys
sys.path.append('..')
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import torch
import torch.nn as nn
import numpy as np

from test_reconst_v2 import ConstructionBezier
from test_loss_define_v2 import ContourChamferLoss, TipDistanceLoss, GenerateRefData

class CatheterOptimizeModel(nn.Module): 
    '''
    This class is used to optimize the catheter parameters.
    '''
    def __init__(self, p_start, para_init, image_ref, gpu_or_cpu, img_save_path): 
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
        self.img_save_path = img_save_path

        ###========================================================
        ### 2) SETTING UP LOSS FUNCTIONS
        ###========================================================
        self.contour_chamfer_loss = ContourChamferLoss(device=gpu_or_cpu)
        self.contour_chamfer_loss.to(gpu_or_cpu)
    
        self.tip_distance_loss = TipDistanceLoss(device=gpu_or_cpu)
        self.tip_distance_loss.to(gpu_or_cpu)

        # Declare self.tip_euclidean_distance_loss as a variable that'll hold a single numpy scalar value
        self.tip_euclidean_distance_loss = None

        ###========================================================
        ### 3) SETTING UP CURVE PARAMETERS
        ###========================================================

        self.p_start = p_start.to(gpu_or_cpu).detach()
        self.para_init = nn.Parameter(torch.from_numpy(para_init).to(gpu_or_cpu),
                                      requires_grad=True)

        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Generate reference data, so you don't need to generate it in every forward pass
        self.generate_ref_data = GenerateRefData(self.image_ref)
        ref_catheter_contour = self.generate_ref_data.get_raw_contour()
        self.register_buffer('ref_catheter_contour', ref_catheter_contour)
        ref_catheter_centerline = self.generate_ref_data.get_raw_centerline()
        self.register_buffer('ref_catheter_centerline', ref_catheter_centerline)
        
        self.gpu_or_cpu = gpu_or_cpu


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
        build_bezier = ConstructionBezier()
        build_bezier.to(self.gpu_or_cpu)
        build_bezier.loadRawImage(self.img_save_path)
        
        build_bezier.getBezierCurveCylinder(self.p_start, self.para_init)
        # Get 2d projected Bezier Cylinder mesh points
        build_bezier.getCylinderMeshProjImg()
        # Get 2d projected Bezier centerline (position) points
        build_bezier.getBezierProjImg()
        build_bezier.draw2DCylinderImage(self.image_ref, save_img_path)

        # TODO: add function to save image to file

        ###========================================================
        ### 4) Compute Loss using various Loss Functions
        ###========================================================
        loss_contour = self.contour_chamfer_loss(build_bezier.bezier_proj_img.to(self.gpu_or_cpu), self.ref_catheter_contour.to(self.gpu_or_cpu))
        loss_tip_distance, self.tip_euclidean_distance_loss = self.tip_distance_loss(build_bezier.bezier_proj_centerline_img.to(self.gpu_or_cpu), self.ref_catheter_centerline.to(self.gpu_or_cpu))

        weight = torch.tensor([1.0, 3.0])
        loss = loss_contour * weight[0] + loss_tip_distance * weight[1]
        
        # TODO: Plot the loss
        self.loss = loss
        self.loss_contour = loss_contour
        self.loss_tip_distance = loss_tip_distance

        return loss