""" 
The version of 3 loss functions that works well.
No issues about gradient computation complexity accumulates with the number of iterations
"""

"""
File that runs the actual reconstruction optimizer forward pass. 
It should compute 3 loss functions: contour chamfer loss, tip distance 
loss, and motion model loss. In its forward pass, it will call upon 
the construction_bezier and motion_catheter scripts to build its 
catheters. It will be optimizing the parameter para_init. 
"""

import sys
sys.path.append('..')
sys.path.insert(1, 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts')

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from construction_bezier import ConstructionBezier
from loss_define import (
    ContourChamferLoss, 
    TipDistanceLoss, 
    ImageContourChamferLoss, 
    GenerateRefData
)

from catheter_motion_tensor import CatheterMotion
from utils import *

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2'
dataset_folder = "gt_dataset3"
test_idx = datetime.now().strftime("%m%d%H%M")
result_folder = f"test_imgs/results_complete_{test_idx}"
rendered_imgs_folder = result_folder + '/' + f"rendered_imgs"


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

    def forward(self, save_img_path): 
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
    


# p_start and para_init used for SRC presentation
# ground truth: [2.15634587e-02 -6.05764476e-04  5.16317712e-01  1.65068886e-01 -2.39781477e-01  9.49010349e-01]
p_start = torch.tensor([0.02, 0.002, 0.000001]) # 0 here will cause NaN in draw2DCylinderImage, pTip

# para_init = np.array([0.034, -0.003, 0.526, 0.13, -0.24, 0.6],
#                  dtype=np.float32) #1
para_init = np.array([0.034, -0.01, 0.536, 0.2, -0.37, 0.6],
                    dtype=np.float32)

gt_name = 'gt_6_0.0006_-0.0010_0.2_0.01'
case_naming = scripts_path + '/' + dataset_folder + '/' + gt_name
# case_naming = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/gt_dataset3/gt_6_0.0006_-0.0010_0.2_0.01'
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
img_ref_binary = process_image(img_save_path)

# Declare loss history lists to keep track of loss values
proj_end_effector_loss_history = []
d3d_end_effector_loss_history = []
loss_history = []
loss_contour_history = []
loss_tip_distance_history = []
loss_motion_model_history = []

# Ground Truth parameters for catheter used in SRC presentation
para_gt_np = read_gt_params(cc_specs_path)
para_gt = torch.tensor(para_gt_np, dtype=torch.float, device=gpu_or_cpu, requires_grad=False)
end_effector_gt = para_gt[3:6]

# folder_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/gt_dataset3/'
folder_path = scripts_path + '/' + dataset_folder + '/'
# image_save_path_list = [
# folder_path + 'gt_5_0.0006_-0.0008_0.2_0.01.png',
# folder_path + 'gt_4_0.0006_-0.0006_0.2_0.01.png',
# folder_path + 'gt_3_0.0006_-0.0004_0.2_0.01.png',
# folder_path + 'gt_2_0.0006_-0.0002_0.2_0.01.png'
# ]

image_save_path_list = [
folder_path + 'gt_4_0.0006_-0.0006_0.2_0.01.png',
folder_path + 'gt_2_0.0006_-0.0002_0.2_0.01.png'
]

past_frames_list = []
for path in image_save_path_list:
    past_frames_list.append(process_image(path))

# delta_u_list = [[0, -0.0002], [0, -0.0002], [0, -0.0002], [0, -0.0002]]
# delta_u_list = torch.tensor([[0, -0.0002], [0, -0.0002], [0, -0.0002], [0, -0.0002]])
delta_u_list = torch.tensor([[0, -0.0004], [0, -0.0004]])




###========================================================
### 3) SET UP AND RUN OPTIMIZATION MODEL
###========================================================
catheter_optimize_model = ReconstructionOptimizer(p_start, para_init, img_ref_binary, gpu_or_cpu, past_frames_list, delta_u_list, img_save_path, image_save_path_list).to(gpu_or_cpu)

print("Model Parameters:")
for name, param in catheter_optimize_model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

learning_rate = 3e-2
optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=learning_rate)

# Run the optimization loop
num_iterations = 2000
loop = tqdm(range(num_iterations))
for loop_id in loop:
    # print("\n================================================================================================================")
    # print("loop_id: ", loop_id)
        
    # save_img_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2/test_imgs/rendered_imgs_new3/' \
    #     + 'render_' + str(loop_id) + '.jpg'
        
    save_img_path = scripts_path + '/' + rendered_imgs_folder + '/' + 'render_' + str(loop_id) + '.jpg'

    # pdb.set_trace()

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Run the forward pass
    loss = catheter_optimize_model(save_img_path)

    # Print gradients for all parameters before backward pass
    # print("Gradients BEFORE BACKWARD PASS:")
    # for name, param in catheter_optimize_model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")  # Print the norm of the gradient
    #     else:
    #         print(f"{name}: No gradient computed")

    # Run the backward pass
    # loss.backward(retain_graph=True)
    loss.backward()

    # Print gradients for all parameters after backward pass
    # print("Gradients AFTER BACKWARD PASS:")
    # for name, param in catheter_optimize_model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")
    #     else:
    #         print(f"{name}: No gradient computed")
            
    # end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
    proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
    d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
    loss_history.append(loss.item())
    loss_contour_history.append(catheter_optimize_model.loss_contour.item())
    loss_tip_distance_history.append(catheter_optimize_model.loss_tip_distance.item())
    loss_motion_model_history.append(catheter_optimize_model.motion_model_loss.item())

    # Update the parameters
    optimizer.step()


    # Print and inspect the updated parameters
    # for name, param in catheter_optimize_model.named_parameters():
    #     print(f"Parameter: {name}, Updated Value: {param.data}")


    # # end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
    # proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
    # d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())

    # Update the progress bar
    loop.set_description('Optimizing')

    # Update the loss
    loop.set_postfix(loss=loss.item())

    # print("Loss: ", loss.item())
    if loop_id % 100 == 0 and loop_id != 0:
        filename = f"bezier_params_{loop_id}.npy"
        full_path = scripts_path + '/' + result_folder + '/' + filename
        para = catheter_optimize_model.para_init.data.cpu().numpy()
        np.save(full_path, para)

# end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
loss_history.append(loss.item())
loss_contour_history.append(catheter_optimize_model.loss_contour.item())
loss_tip_distance_history.append(catheter_optimize_model.loss_tip_distance.item())
loss_motion_model_history.append(catheter_optimize_model.motion_model_loss.item())

filename = "parameters.txt"
full_path = scripts_path + '/' + result_folder + '/' + filename
with open(full_path, 'w') as file:
    file.write(f"p_start = {p_start.numpy().tolist()}\n")
    file.write(f"para_init = {para_init.tolist()}\n")
    file.write(f"gt_path = {img_save_path}\n")
    file.write(f"learning_rate = {learning_rate}\n")
    file.write(f"loss_history = {loss_history}\n")
    file.write(f"loss_contour_history = {loss_contour_history}\n")
    file.write(f"loss_tip_distance_history = {loss_tip_distance_history}\n")
    file.write(f"loss_motion_model_history = {loss_motion_model_history}\n")
    
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
# ax1.set_ylim([0, 80])
ax1.set_ylim(bottom=0)
ax1.grid(True)

filename = "2D_tip_loss.png"
full_path = scripts_path + '/' + result_folder + '/' + filename
if not os.path.exists(os.path.dirname(full_path)):
    os.makedirs(os.path.dirname(full_path))
plt.savefig(full_path)

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
ax2.set_ylim(bottom=0)
# ax2.set_ylim([0, 0.05])
ax2.grid(True)

filename = "3D_tip_loss.png"
full_path = scripts_path + '/' + result_folder + '/' + filename
plt.savefig(full_path)

plt.show()

p_start_np = p_start.numpy()
result = catheter_optimize_model.para_init.data.cpu().numpy()
control_points = np.vstack([p_start_np, result.reshape(2, 3)])
control_points_gt = np.vstack([p_start_np, para_gt_np.reshape(2, 3)])
control_points_init = np.vstack([p_start_np, para_init.reshape(2, 3)])

filename = "bezier_params.npz"
full_path = scripts_path + '/' + result_folder + '/' + filename
np.savez(full_path, control_points=control_points, control_points_gt=control_points_gt, control_points_init=control_points_init)

# Generate the Bezier curve
curve = bezier_curve_3d(control_points)
curve_gt = bezier_curve_3d(control_points_gt)
curve_init = bezier_curve_3d(control_points_init)


# Plotting the Bezier curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制第一条贝塞尔曲线及其控制点
ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--')
ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Optimized Result')

# 绘制第二条贝塞尔曲线及其控制点
ax.plot(control_points_gt[:, 0], control_points_gt[:, 1], control_points_gt[:, 2], 'bo--')
ax.plot(curve_gt[:, 0], curve_gt[:, 1], curve_gt[:, 2], 'b-', label='Ground Truth')

ax.plot(control_points_init[:, 0], control_points_init[:, 1], control_points_init[:, 2], 'go--')
ax.plot(curve_init[:, 0], curve_init[:, 1], curve_init[:, 2], 'g-', label='Initial Guess')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Optimization Result')
ax.legend()

filename = "visualization.png"
full_path = scripts_path + '/' + result_folder + '/' + filename
plt.savefig(full_path)

plt.show()




iterations_x_axis_loss = list(range(len(loss_history)))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.suptitle('Total Loss History')
ax2.plot(iterations_x_axis_loss, loss_history)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Total Loss')
ax2.set_xlim([0, len(loss_history)])
ax2.set_ylim(bottom=0)
# ax2.set_ylim([0, 0.05])
ax2.grid(True)

filename = "total_loss.png"
full_path = scripts_path + '/' + result_folder + '/' + filename
plt.savefig(full_path)

plt.show()



iterations_x_axis_loss_contour = list(range(len(loss_contour_history)))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.suptitle('Contour Loss History')
ax2.plot(iterations_x_axis_loss_contour, loss_contour_history)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Contour Loss')
ax2.set_xlim([0, len(loss_contour_history)])
ax2.set_ylim(bottom=0)
# ax2.set_ylim([0, 0.05])
ax2.grid(True)

filename = "contour_loss.png"
full_path = scripts_path + '/' + result_folder + '/' + filename
plt.savefig(full_path)

plt.show()



iterations_x_axis_loss_tip = list(range(len(loss_tip_distance_history)))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.suptitle('Tip Loss History')
ax2.plot(iterations_x_axis_loss_tip, loss_tip_distance_history)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Tip Loss')
ax2.set_xlim([0, len(loss_tip_distance_history)])
ax2.set_ylim(bottom=0)
# ax2.set_ylim([0, 0.05])
ax2.grid(True)

filename = "tip_loss.png"
full_path = scripts_path + '/' + result_folder + '/' + filename
plt.savefig(full_path)

plt.show()


iterations_x_axis_loss_mm = list(range(len(loss_motion_model_history)))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.suptitle('Motion Model Loss History')
ax2.plot(iterations_x_axis_loss_mm, loss_motion_model_history)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Motion Model Loss')
ax2.set_xlim([0, len(loss_motion_model_history)])
ax2.set_ylim(bottom=0)
# ax2.set_ylim([0, 0.05])
ax2.grid(True)

filename = "motion_model_loss.png"
full_path = scripts_path + '/' + result_folder + '/' + filename
plt.savefig(full_path)

plt.show()