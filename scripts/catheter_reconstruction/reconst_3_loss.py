"""
Reconstruction optimizer with 3 loss functions, used in control pipeline.
"""

import sys
sys.path.append('..')
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from collections import deque
import pickle
import os

from catheter_reconstruction.construction_bezier import ConstructionBezier
from catheter_reconstruction.loss_define import (
    ContourChamferLoss, 
    TipDistanceLoss, 
    GenerateRefData
)
from catheter_reconstruction.catheter_motion_tensor import CatheterMotion
from catheter_reconstruction.utils import *

class CatheterOptimizeModel(nn.Module): 

    def __init__(self, p_start, para_init, image_ref, gpu_or_cpu, past_frames_list, delta_u_list, img_save_path, image_save_path_list, l=0.2): 
        '''
        This class is used to optimize the catheter parameters.

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
        
        self.img_save_path = img_save_path

        self.contour_chamfer_loss = ContourChamferLoss(device=gpu_or_cpu)
        self.contour_chamfer_loss.to(gpu_or_cpu)
        self.tip_distance_loss = TipDistanceLoss(device=gpu_or_cpu)
        self.tip_distance_loss.to(gpu_or_cpu)
        
        # Declare self.tip_euclidean_distance_loss as a variable that'll hold a single numpy scalar value
        self.tip_euclidean_distance_loss = None
        self.tip_loss = None

        self.p_start = p_start.to(gpu_or_cpu).detach()
        self.para_init = nn.Parameter(para_init.to(gpu_or_cpu), requires_grad=True)
        
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Generate reference data, so you don't need to generate it in every forward pass
        self.generate_ref_data = GenerateRefData(self.image_ref)
        ref_catheter_contour = self.generate_ref_data.get_raw_contour()
        self.ref_catheter_contour = ref_catheter_contour
        ref_catheter_centerline = self.generate_ref_data.get_raw_centerline()
        self.ref_catheter_centerline = ref_catheter_centerline
        
        self.delta_u_list = delta_u_list.to(gpu_or_cpu)
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
        self.contour_list = contour_list
        self.centerline_list = centerline_list
        
        self.gpu_or_cpu = gpu_or_cpu
        self.l = l

    def forward(self, save_img_path=None, test=False): 
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
        
        if test:
            build_bezier.draw2DCylinderImage(self.image_ref, save_img_path + 'current.png')

        loss_contour = self.contour_chamfer_loss(build_bezier.bezier_proj_img.to(self.gpu_or_cpu), self.ref_catheter_contour.to(self.gpu_or_cpu))

        loss_tip_distance, self.tip_euclidean_distance_loss = self.tip_distance_loss(build_bezier.bezier_proj_centerline_img.to(self.gpu_or_cpu), self.ref_catheter_centerline.to(self.gpu_or_cpu))
        
        catheterMotion = CatheterMotion(self.p_start, self.gpu_or_cpu, l=self.l)
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
            
            if test:
                img_ref_binary = process_image(self.image_save_path_list[i])
                image_ref = torch.from_numpy(img_ref_binary.astype(np.float32))
                construction_bezier.draw2DCylinderImage(image_ref, save_img_path + 'predicted_' + str(i) + '.png')
            
            
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
    
def optimize_3_loss(gt_img_path, gt_specs_path, iteration, result_save_path, para_init, gt_img_path_list, delta_u_list, l, learning_rate = 1e-2, max_iterations = 1500, show=False):
    '''
    Main function to set up optimzer model and run the optimization loop
    Args:
        gt_img_path (str): path to the ground truth image
        gt_specs_path (str): path to the ground truth specs
        iteration (int): iteration number
        result_save_path (str): path to save the optimization results
        para_init (tensor, length 6): initial guess
    '''
    
    # print(f"Begin optimization for iteration {iteration}")
    # print("gt_img_path: ", gt_img_path)
    # print("gt_specs_path: ", gt_specs_path)
    # print("result_save_path: ", result_save_path)
    # print("gt_img_path_list: ", gt_img_path_list)
    # print("delta_u_list: ", delta_u_list)
    print("learning_rate: ", learning_rate)
    
    # Create the folder to save the optimization results
    result_save_path = os.path.join(result_save_path, f'iter_{iteration}')
    if result_save_path is not None:
        if not os.path.exists(os.path.dirname(result_save_path)):
            os.makedirs(os.path.dirname(result_save_path))
            
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
    p_start = torch.tensor(np.load(gt_specs_path)[0, 0, :] + 0.000001, dtype=torch.float) # 0 here will cause NaN in draw2DCylinderImage, pTip

    img_ref_binary = process_image(gt_img_path)

    # Declare loss history lists to keep track of loss values
    proj_end_effector_loss_history = []
    d3d_end_effector_loss_history = []
    d3d_mid_control_point_loss_history = []
    loss_history = []
    loss_contour_history = []
    loss_tip_distance_history = []
    loss_motion_model_history = []

    # Ground Truth parameters for catheter used in SRC presentation
    para_gt_np = read_gt_params(gt_specs_path)
    para_gt = torch.tensor(para_gt_np, dtype=torch.float, device=gpu_or_cpu, requires_grad=False)
    end_effector_gt = para_gt[3:6]
    mid_control_point_gt = para_gt[:3]
    
    past_frames_list = []
    for path in gt_img_path_list:
        past_frames_list.append(process_image(path))
        
    delta_u_list = torch.tensor(delta_u_list, dtype=torch.float)    

    ###========================================================
    ### 3) SET UP AND RUN OPTIMIZATION MODEL
    ###========================================================
    catheter_optimize_model = CatheterOptimizeModel(p_start, para_init, img_ref_binary, gpu_or_cpu, past_frames_list, delta_u_list, gt_img_path, gt_img_path_list, l=l).to(gpu_or_cpu)
    
    catheter_optimize_model_test = CatheterOptimizeModel(p_start, para_gt, img_ref_binary, gpu_or_cpu, past_frames_list, delta_u_list, gt_img_path, gt_img_path_list, l=l).to(gpu_or_cpu)
    save_img_path = result_save_path + '/' + 'rendered_imgs' + '/'
    catheter_optimize_model_test.forward(save_img_path, test=True)

    optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    convergence_window_size = 5
    convergence_threshold = 1e-2 
    loss_queue = deque(maxlen=convergence_window_size)
    
    # Run the optimization loop
    loop = tqdm(range(max_iterations))
    for loop_id in loop:
            
        save_img_path = result_save_path + '/' + 'rendered_imgs' + '/' + 'render_' + str(loop_id) + '.jpg'

        optimizer.zero_grad()

        # # Run the forward pass
        # loss = catheter_optimize_model(save_img_path)
        
        # Run the forward pass
        if loop_id % 10 == 0: # save rendered image every 10 iterations
            loss = catheter_optimize_model(save_img_path)
        else:
            loss = catheter_optimize_model()

        proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
        d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
        d3d_mid_control_point_loss_history.append(torch.norm((catheter_optimize_model.para_init[:3] - mid_control_point_gt), p=2).item())
        loss_history.append(loss.item())
        loss_contour_history.append(catheter_optimize_model.loss_contour.item())
        loss_tip_distance_history.append(catheter_optimize_model.loss_tip_distance.item())
        loss_motion_model_history.append(catheter_optimize_model.motion_model_loss.item())

        # Run the backward pass
        loss.backward()
        
        # Update the parameters
        optimizer.step()

        # Update the progress bar
        loop.set_description('Optimizing')

        # Update the loss
        loop.set_postfix(loss=loss.item())
        
        # Save the optimized bezier control points every 100 iterations
        
        
        loss_queue.append(loss.item())

        # Check for convergence
        if len(loss_queue) == convergence_window_size:
            max_loss = max(loss_queue)
            min_loss = min(loss_queue)
            if max_loss - min_loss < convergence_threshold:
                print(f"Converged at iteration {loop_id}")
                break


    proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
    d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
    d3d_mid_control_point_loss_history.append(torch.norm((catheter_optimize_model.para_init[:3] - mid_control_point_gt), p=2).item())
    loss_history.append(loss.item())
    loss_contour_history.append(catheter_optimize_model.loss_contour.item())
    loss_tip_distance_history.append(catheter_optimize_model.loss_tip_distance.item())
    loss_motion_model_history.append(catheter_optimize_model.motion_model_loss.item())

    filename = "parameters.txt"
    full_path = result_save_path + '/' + filename
    with open(full_path, 'w') as file:
        file.write(f"p_start = {p_start.numpy().tolist()}\n")
        file.write(f"para_init = {para_init.tolist()}\n")
        file.write(f"gt_path = {gt_img_path}\n")
        file.write(f"gt_past_frames = {gt_img_path_list}\n")
        file.write(f"learning_rate = {learning_rate}\n")
        file.write(f"loss_history = {loss_history}\n")
        file.write(f"loss_contour_history = {loss_contour_history}\n")
        file.write(f"loss_tip_distance_history = {loss_tip_distance_history}\n")
        file.write(f"loss_motion_model_history = {loss_motion_model_history}\n")
        
    data = {'proj_end_effector_loss_history': proj_end_effector_loss_history,
            'd3d_end_effector_loss_history': d3d_end_effector_loss_history,
            'd3d_mid_control_point_loss_history': d3d_mid_control_point_loss_history,
            'loss_history': loss_history, 
            'loss_contour_history': loss_contour_history,
            'loss_tip_distance_history': loss_tip_distance_history,
            'loss_motion_model_history': loss_motion_model_history, 
            'para_init': para_init,
            'p_start': p_start.numpy(),
            'gt_path': gt_img_path,
            'learning_rate': learning_rate
            }
    full_path = result_save_path + '/' + "data.pkl"
    with open(full_path, 'wb') as f:
        pickle.dump(data, f)

    full_path = result_save_path + '/' + "2D_tip_loss.png"
    plot_2d_end_effector_loss(proj_end_effector_loss_history, full_path, show=show)  
    
    
    initial_loss = d3d_end_effector_loss_history[0]
    final_loss = d3d_end_effector_loss_history[-1]
    loss_decrease_percentage = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"3D end effector - Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}, Loss Decrease Percentage: {loss_decrease_percentage:.2f}%")
    
    full_path = result_save_path + '/' + "3D_tip_loss.png"  
    plot_3d_end_effector_loss(d3d_end_effector_loss_history, full_path, show=show)
    
    
    initial_loss = d3d_mid_control_point_loss_history[0]
    final_loss = d3d_mid_control_point_loss_history[-1]
    loss_decrease_percentage = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"3D middle control point - Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}, Loss Decrease Percentage: {loss_decrease_percentage:.2f}%")
    
    full_path = result_save_path + '/' + "3D_shape_loss.png" 
    plot_3d_mid_control_point_loss(d3d_mid_control_point_loss_history, full_path, show=show)
    

    p_start_np = p_start.numpy()
    result = catheter_optimize_model.para_init.data.detach().cpu().numpy()
    
    control_points = np.vstack([p_start_np, result.reshape(2, 3)])
    control_points_gt = np.vstack([p_start_np, para_gt_np.reshape(2, 3)])
    control_points_init = np.vstack([p_start_np, para_init.numpy().reshape(2, 3)])
    
    filename = "bezier_params_result.npy"
    full_path = result_save_path + '/' + filename
    np.save(full_path, control_points)

    full_path = result_save_path + '/' + "bezier_params.npz"
    np.savez(full_path, control_points=control_points, control_points_gt=control_points_gt, control_points_init=control_points_init)

    full_path = result_save_path + '/' + "visualization.png"
    plot_3D_bezier_curve(control_points=control_points, control_points_gt=control_points_gt, control_points_init=control_points_init, save_path=full_path, equal=False, show=show)
    
    full_path = result_save_path + '/' + "total_loss.png"
    plot_total_loss(loss_history, full_path, log_scale=True, show=show)

    full_path = result_save_path + '/' + "contour_loss.png"
    plot_contour_loss(loss_contour_history, full_path, log_scale=True, show=show)
    
    full_path = result_save_path + '/' + "tip_loss.png"
    plot_tip_loss(loss_tip_distance_history, full_path, log_scale=True, show=show)
    
    full_path = result_save_path + '/' + "motion_model_loss.png"
    plot_motion_model_loss(loss_motion_model_history, full_path, log_scale=True, show=show)
    
    return result