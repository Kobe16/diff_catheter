"""
File to rebuild past frame catheters, using inverse and forward kinematics. 
In the end, it should output a reconstructed image of a catheter from k 
frames ago. It will call upon bezier_interspace_transforms for its calculations. 
"""
import sys
import os
import torch

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from catheter_reconstruction.bezier_interspace_transforms_tensor import *

class CatheterMotion(): 

    def __init__(self, p_0, gpu_or_cpu, l=0.2, r=0.01):
        """
        Args:
            p_0 ((3,) torch tensor): start point of catheter
            l (float): length of catheter
            r (float): cross section radius of catheter
        """
        self.gpu_or_cpu = gpu_or_cpu
        
        self.p_0 = torch.cat((p_0, torch.tensor([1.0]).to(gpu_or_cpu))).to(gpu_or_cpu)
        self.l = l
        self.r = r

    def past_frames_prediction(self, delta_u_list, p_init):
        """
        delta_u_list: start from delta_u_(n-1)
        
        Args:
            delta_u_list (torch tensor): shape (n, 2)
            p_init (torch tensor): initial points, concatenation of p1 and p2, shape (6,)
        """
        
        p_1 = torch.cat((p_init[:3], torch.tensor([1.0]).to(self.gpu_or_cpu))).to(self.gpu_or_cpu)
        p_2 = torch.cat((p_init[3:], torch.tensor([1.0]).to(self.gpu_or_cpu))).to(self.gpu_or_cpu)
        # print("p_init, p_1, p_2.requires_grad: ", p_init.requires_grad, p_1.requires_grad, p_2.requires_grad)

        ux, uy = bezier_control_points_to_tendon_disp(self.p_0, p_1, p_2, self.l, self.r, self.gpu_or_cpu)
        # print("ux, uy.requires_grad: ", ux.requires_grad, uy.requires_grad)

        # delta_u_cumulative = torch.tensor([ux, uy]).to(self.gpu_or_cpu)
        delta_u_cumulative = torch.stack([ux, uy]).to(self.gpu_or_cpu)
        # print("delta_u_cumulative.requires_grad: ", delta_u_cumulative.requires_grad)
        bezier_control_points = []

        for delta_u in delta_u_list:
            # delta_u_cumulative -= delta_u
            delta_u_cumulative = delta_u_cumulative - delta_u
            # print("delta_u_cumulative.requires_grad: ", delta_u_cumulative.requires_grad)
            p_1_new, p_2_new = tendon_disp_to_bezier_control_points(delta_u_cumulative[0], delta_u_cumulative[1], self.l, self.r, self.p_0, self.gpu_or_cpu)
            # print("p_1_new, p_2_new.requires_grad: ", p_1_new.requires_grad, p_2_new.requires_grad)
            bezier_points = torch.cat((p_1_new[:-1], p_2_new[:-1]))
            bezier_control_points.append(bezier_points)

        return torch.stack(bezier_control_points)
