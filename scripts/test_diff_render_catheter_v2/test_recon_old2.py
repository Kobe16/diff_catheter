"""
The version of 2 loss functinos that works well
"""

'''
File used to run optimization on the catheter parameters.
Compute 2 loss functions: contour loss and tip distance loss. 
Uses Adam optimizer. 
'''
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
import cv2

from test_reconst_v2 import ConstructionBezier
from test_loss_define_v2 import ChamferLossWholeImage, ContourChamferLoss, \
    TipChamferLoss, BoundaryPointChamferLoss, TipDistanceLoss, BoundaryPointDistanceLoss, \
    GenerateRefData

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2'
dataset_folder = "gt_dataset3"
test_idx = datetime.now().strftime("%m%d%H%M")
result_folder = f"test_imgs/results_old_{test_idx}"
rendered_imgs_folder = result_folder + '/' + f"rendered_imgs"



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
        # self.build_bezier = ConstructionBezier()
        # self.build_bezier.to(gpu_or_cpu)
        # self.build_bezier.loadRawImage(img_save_path)
        self.img_save_path = img_save_path

        ###========================================================
        ### 2) SETTING UP LOSS FUNCTIONS
        ###========================================================
        # self.chamfer_loss_whole_image = ChamferLossWholeImage(device=gpu_or_cpu)
        # self.chamfer_loss_whole_image.to(gpu_or_cpu)
        self.contour_chamfer_loss = ContourChamferLoss(device=gpu_or_cpu)
        self.contour_chamfer_loss.to(gpu_or_cpu)
        # self.tip_chamfer_loss = TipChamferLoss(device=gpu_or_cpu)
        # self.tip_chamfer_loss.to(gpu_or_cpu)
        # self.boundary_point_chamfer_loss = BoundaryPointChamferLoss(device=gpu_or_cpu)
        # self.boundary_point_chamfer_loss.to(gpu_or_cpu)
        self.tip_distance_loss = TipDistanceLoss(device=gpu_or_cpu)
        self.tip_distance_loss.to(gpu_or_cpu)
        # self.boundary_point_distance_loss = BoundaryPointDistanceLoss(device=gpu_or_cpu)
        # self.boundary_point_distance_loss.to(gpu_or_cpu)

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

        # self.para_init = nn.Parameter(torch.from_numpy(
        #     np.array([0.0365, 0.0036,  0.1202,  0.0056, -0.0166, 0.1645],
        #              dtype=np.float32)).to(gpu_or_cpu),
        #                               requires_grad=True)
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
        build_bezier.to(gpu_or_cpu)
        build_bezier.loadRawImage(self.img_save_path)
        
        
        build_bezier.getBezierCurveCylinder(self.p_start, self.para_init)

        # cylinder_mesh_points = self.build_bezier.cylinder_mesh_points
        # print("cylinder_mesh_points max value: ", torch.max(cylinder_mesh_points))

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0, 0)
        # loss = torch.mean(torch.norm(cylinder_mesh_points, dim=1))

        # Plot 3D Bezier Cylinder mesh points
        # self.build_bezier.plot3dBezierCylinder()

        # Get 2d projected Bezier Cylinder mesh points
        build_bezier.getCylinderMeshProjImg()


        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all the points in cylinder_mesh_points to (0, 0)
        # bezier_proj_img = self.build_bezier.bezier_proj_img
        # print("bezier_proj_img:" , bezier_proj_img)
        # print("Max value in bezier_proj_img: ", torch.max(bezier_proj_img))
        # print("average value in bezier_proj_img", torch.mean(bezier_proj_img))
        # loss = torch.mean(torch.norm(bezier_proj_img, dim=1))

        # Get 2d projected Bezier centerline (position) points
        build_bezier.getBezierProjImg()

        # TEST LOSS TO SEE IF DIFFERENTIABLE UP TILL THIS POINT
        # find average distance of all points in bezier_proj_centerline_img to (0, 0)
        # bezier_proj_centerline_img = self.build_bezier.bezier_proj_centerline_img
        # loss = torch.mean(torch.norm(bezier_proj_centerline_img, dim=1))

        # Plot 2D projected Bezier Cylinder mesh points
        # print("cylinder_mesh_points: ", self.build_bezier.cylinder_mesh_points)
        build_bezier.draw2DCylinderImage(self.image_ref, save_img_path)

        # TODO: add function to save image to file

        ###========================================================
        ### 4) Compute Loss using various Loss Functions
        ###========================================================
        
        loss_contour = self.contour_chamfer_loss(build_bezier.bezier_proj_img.to(gpu_or_cpu), self.ref_catheter_contour.to(gpu_or_cpu))
        # loss_contour = self.contour_chamfer_loss(self.build_bezier.bezier_proj_centerline_img.to(gpu_or_cpu), self.ref_catheter_contour.to(gpu_or_cpu))
        loss_tip_distance, self.tip_euclidean_distance_loss = self.tip_distance_loss(build_bezier.bezier_proj_centerline_img.to(gpu_or_cpu), self.ref_catheter_centerline.to(gpu_or_cpu))


        weight = torch.tensor([1.0, 1.0])
        loss = loss_contour * weight[0] + loss_tip_distance * weight[1]
        # loss = loss_tip_distance
        # print("loss_contour.requires_grad: ", loss_contour.requires_grad)
        # print("loss_tip_distance.requires_grad: ", loss_tip_distance.requires_grad)
        # print("loss.requires_grad: ", loss.requires_grad)
        
        # TODO: Plot the loss
        self.loss = loss
        self.loss_contour = loss_contour
        self.loss_tip_distance = loss_tip_distance

        return loss

def read_gt_params(cc_specs_path):
    """
    The order of gt: [p_start, p_end, c1, c2]
    """
    
    para_gt_np = np.load(cc_specs_path)
    matrix = np.squeeze(para_gt_np)
    c1 = matrix[2]
    c2 = matrix[3]
    p_start = matrix[0]
    p_end = matrix[1]
    p_mid = 3/4 * (c1 + p_end/3)
    p1 = 2*p_mid - 0.5*p_start - 0.5*p_end
    result_vector = np.concatenate((p1, p_end))
    return result_vector

def process_image(img_save_path):
    img_ref_rgb = cv2.imread(img_save_path)
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)
    (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 10, 255, cv2.THRESH_BINARY)
    img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)
    
    return img_ref_binary

# if __name__ == '__main__':
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
# p_start = torch.tensor([0.02, 0.008, 0.054])
p_start = torch.tensor([0.02, 0.002, 0.000001]) # 0 here will cause NaN in draw2DCylinderImage, pTip
# para_init = np.array([0.03, 0.0055,  0.0702,  0.0206, -0.0306, 0.1845],
#                  dtype=np.float32) # 1
# [ 2.15634587e-02 -6.05764476e-04  5.16317712e-01  1.65068886e-01 -2.39781477e-01  9.49010349e-01]
para_init = np.array([0.034, -0.01, 0.536, 0.2, -0.37, 0.6],
                    dtype=np.float32)

# case_naming = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/data/rendered_images/dof2_64/dof2_c48_0.001_-0.005_0.2_0.01'
gt_name = 'gt_6_0.0006_-0.0010_0.2_0.01'
case_naming = scripts_path + '/' + dataset_folder + '/' + gt_name
img_save_path = case_naming + '.png'
cc_specs_path = case_naming + '.npy'
# cc_specs_path = case_naming + '_gt.npy'
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
# img_ref_rgb = cv2.imread(img_save_path)
# img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)
# (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)
img_ref_binary = process_image(img_save_path)

# Declare loss history lists to keep track of loss values
proj_end_effector_loss_history = []
d3d_end_effector_loss_history = []
loss_history = []
loss_contour_history = []
loss_tip_distance_history = []

# Ground Truth parameters for catheter used in SRC presentation
# para_gt_np = np.load(cc_specs_path)[1:3, :].flatten()
para_gt_np = read_gt_params(cc_specs_path)
# para_gt = torch.tensor([0.02003904, 0.0016096, 0.13205799, 0.00489567, -0.03695673, 0.196168896], dtype=torch.float, device=gpu_or_cpu, requires_grad=False)
para_gt = torch.tensor(para_gt_np, dtype=torch.float, device=gpu_or_cpu, requires_grad=False)
end_effector_gt = para_gt[3:6]


###========================================================
### 3) SET UP AND RUN OPTIMIZATION MODEL
###========================================================
catheter_optimize_model = CatheterOptimizeModel(p_start, para_init, img_ref_binary, gpu_or_cpu, img_save_path).to(gpu_or_cpu)

# optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=1e-2)
learning_rate = 3e-2
optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=learning_rate)

# Run the optimization loop
num_iterations = 200
loop = tqdm(range(num_iterations))
for loop_id in loop:     
    save_img_path = scripts_path + '/' + rendered_imgs_folder + '/' + 'render_' + str(loop_id) + '.jpg'

    optimizer.zero_grad()

    # Run the forward pass
    loss = catheter_optimize_model(save_img_path)

    # Run the backward pass
    # loss.backward(retain_graph=True)
    loss.backward()
    
    # end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
    proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
    d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
    loss_history.append(loss.item())
    loss_contour_history.append(catheter_optimize_model.loss_contour.item())
    loss_tip_distance_history.append(catheter_optimize_model.loss_tip_distance.item())

    # Update the parameters
    optimizer.step()

    # Update the progress bar
    loop.set_description('Optimizing')

    # Update the loss
    loop.set_postfix(loss=loss.item())


proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
loss_history.append(loss.item())
loss_contour_history.append(catheter_optimize_model.loss_contour.item())
loss_tip_distance_history.append(catheter_optimize_model.loss_tip_distance.item())

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

# Given array of values proj_end_effector_loss_history, create plot of loss vs. iterations
iterations_x_axis_proj = list(range(len(proj_end_effector_loss_history)))
# print("proj_end_effector_loss_history: ", proj_end_effector_loss_history)
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

def bezier_curve_3d(control_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 3))

    for i in range(num_points):
        curve[i] = (1 - t[i]) ** 2 * control_points[0] + \
                2 * (1 - t[i]) * t[i] * control_points[1] + \
                t[i] ** 2 * control_points[2]

    return curve

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
print("loss_history: ", loss_history)
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