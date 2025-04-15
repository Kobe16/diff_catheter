from tqdm.auto import tqdm
import torch
from datetime import datetime
from collections import deque
import pickle
import os

from recon_old import CatheterOptimizeModel
from utils import *

def main(scripts_path, dataset_folder, gt_name, rendered_imgs_folder, result_folder, para_init, learning_rate = 3e-2, max_iterations = 200):
    '''
    Main function to set up optimzer model and run the optimization loop
    '''
    
    path = scripts_path + '/' + result_folder
    if path is not None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
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
    case_naming = scripts_path + '/' + dataset_folder + '/' + gt_name
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0
    
    p_start = torch.tensor(np.load(cc_specs_path)[0, 0, :] + 0.000001, dtype=torch.float) # 0 here will cause NaN in draw2DCylinderImage, pTip

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
    d3d_mid_control_point_loss_history = []
    loss_history = []
    loss_contour_history = []
    loss_tip_distance_history = []

    # Ground Truth parameters for catheter used in SRC presentation
    para_gt_np = read_gt_params(cc_specs_path)
    para_gt = torch.tensor(para_gt_np, dtype=torch.float, device=gpu_or_cpu, requires_grad=False)
    end_effector_gt = para_gt[3:6]
    mid_control_point_gt = para_gt[:3]    

    ###========================================================
    ### 3) SET UP AND RUN OPTIMIZATION MODEL
    ###========================================================
    catheter_optimize_model = CatheterOptimizeModel(p_start, para_init, img_ref_binary, gpu_or_cpu, img_save_path).to(gpu_or_cpu)

    optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    convergence_window_size = 10
    convergence_threshold = 5e-3 
    loss_queue = deque(maxlen=convergence_window_size)
    
    # Run the optimization loop
    loop = tqdm(range(max_iterations))
    for loop_id in loop:
             
        save_img_path = scripts_path + '/' + rendered_imgs_folder + '/' + 'render_' + str(loop_id) + '.jpg'

        optimizer.zero_grad()

        # Run the forward pass
        loss = catheter_optimize_model(save_img_path)

        # Run the backward pass
        loss.backward()
        
        # end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
        proj_end_effector_loss_history.append(catheter_optimize_model.tip_euclidean_distance_loss.item())
        d3d_end_effector_loss_history.append(torch.norm((catheter_optimize_model.para_init[3:6] - end_effector_gt), p=2).item())
        d3d_mid_control_point_loss_history.append(torch.norm((catheter_optimize_model.para_init[:3] - mid_control_point_gt), p=2).item())
        loss_history.append(loss.item())
        loss_contour_history.append(catheter_optimize_model.loss_contour.item())
        loss_tip_distance_history.append(catheter_optimize_model.loss_tip_distance.item())

        # Update the parameters
        optimizer.step()

        # Update the progress bar
        loop.set_description('Optimizing')

        # Update the loss
        loop.set_postfix(loss=loss.item())
        
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
        
    data = {'proj_end_effector_loss_history': proj_end_effector_loss_history,
            'd3d_end_effector_loss_history': d3d_end_effector_loss_history,
            'd3d_mid_control_point_loss_history': d3d_mid_control_point_loss_history,
            'loss_history': loss_history, 
            'loss_contour_history': loss_contour_history,
            'loss_tip_distance_history': loss_tip_distance_history, 
            'para_init': para_init,
            'p_start': p_start.numpy(),
            'gt_path': img_save_path,
            'learning_rate': learning_rate
            }
    full_path = scripts_path + '/' + result_folder + '/' + "data.pkl"
    with open(full_path, 'wb') as f:
        pickle.dump(data, f)

    full_path = scripts_path + '/' + result_folder + '/' + "2D_tip_loss.png"
    plot_2d_end_effector_loss(proj_end_effector_loss_history, full_path)  
    
    
    initial_loss = d3d_end_effector_loss_history[0]
    final_loss = d3d_end_effector_loss_history[-1]
    loss_decrease_percentage = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"3D end effector - Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}, Loss Decrease Percentage: {loss_decrease_percentage:.2f}%")
    
    full_path = scripts_path + '/' + result_folder + '/' + "3D_tip_loss.png"  
    plot_3d_end_effector_loss(d3d_end_effector_loss_history, full_path)
    
    
    initial_loss = d3d_mid_control_point_loss_history[0]
    final_loss = d3d_mid_control_point_loss_history[-1]
    loss_decrease_percentage = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"3D shape - Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}, Loss Decrease Percentage: {loss_decrease_percentage:.2f}%")
    
    full_path = scripts_path + '/' + result_folder + '/' + "3D_shape_loss.png" 
    plot_3d_mid_control_point_loss(d3d_mid_control_point_loss_history, full_path)
    

    p_start_np = p_start.numpy()
    result = catheter_optimize_model.para_init.data.cpu().numpy()
    
    control_points = np.vstack([p_start_np, result.reshape(2, 3)])
    control_points_gt = np.vstack([p_start_np, para_gt_np.reshape(2, 3)])
    control_points_init = np.vstack([p_start_np, para_init.reshape(2, 3)])
    
    filename = "bezier_params_result.npy"
    full_path = scripts_path + '/' + result_folder + '/' + filename
    np.save(full_path, control_points)

    full_path = scripts_path + '/' + result_folder + '/' + "bezier_params.npz"
    np.savez(full_path, control_points=control_points, control_points_gt=control_points_gt, control_points_init=control_points_init)

    full_path = scripts_path + '/' + result_folder + '/' + "visualization.png"
    plot_3D_bezier_curve(control_points=control_points, control_points_gt=control_points_gt, control_points_init=control_points_init, save_path=full_path, equal=False)
    
    full_path = scripts_path + '/' + result_folder + '/' + "total_loss.png"
    plot_total_loss(loss_history, full_path, log_scale=True)

    full_path = scripts_path + '/' + result_folder + '/' + "contour_loss.png"
    plot_contour_loss(loss_contour_history, full_path, log_scale=True)
    
    full_path = scripts_path + '/' + result_folder + '/' + "tip_loss.png"
    plot_tip_loss(loss_tip_distance_history, full_path, log_scale=True)
    

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2'
dataset_folder = "gt_dataset4"

# case 1
# gt_name = 'gt_35_-0.0008_0.0008_0.2_0.01'
# para_init = np.array([0.02, -0.01, 0.436, -0.19, 0.13, 0.63], dtype=np.float32)

# case 2
# gt_name = 'gt_11_-0.0012_0.0008_0.2_0.01'
# para_init = np.array([0.02, -0.01, 0.406, -0.19, 0.18, 0.65], dtype=np.float32)
# learning_rate = 5e-2

# case 3
# gt_name = 'gt_19_-0.0010_0.0000_0.2_0.01'
# para_init = np.array([-0.026, -0.012, 0.517, -0.240, -0.019, 0.974], dtype=np.float32)
# learning_rate = 5e-4

# case 4
# gt_name = 'gt_24_-0.0010_0.0010_0.2_0.01'
# para_init = np.array([0.03265457, 0.01057332, 0.4779693, -0.19458942, 0.25682682, 0.85749231], dtype=np.float32)
# learning_rate = 3e-2

# case 5
# gt_name = 'gt_30_-0.0008_-0.0002_0.2_0.01'
# para_init = np.array([0.0442238, -0.0385103, 0.4705782, -0.17881069, 0.00760917, 0.97061609], dtype=np.float32)
# learning_rate = 1e-2

# case 6
# gt_name = 'gt_35_-0.0008_0.0008_0.2_0.01'
# para_init = np.array([-0.02816927, 0.05947484, 0.53783347, -0.20232028, 0.18103059, 0.90563674], dtype=np.float32)
# learning_rate = 5e-3

# case 7
# gt_name = 'gt_42_-0.0006_-0.0002_0.2_0.01'
# para_init = np.array([0.06446165, 0.00630285, 0.5086952, -0.16350525, -0.07417695, 0.9612858], dtype=np.float32)
# learning_rate = 1e-3

# case 8
# gt_name = 'gt_144_0.0010_0.0010_0.2_0.01'
# para_init = np.array([0.04237379, -0.0279111, 0.61584815, 0.27306871, 0.25127924, 0.85242042], dtype=np.float32)
# learning_rate = 1e-2

# case 9
# gt_name = 'gt_138_0.0010_-0.0002_0.2_0.01'
# para_init = np.array([0.00486385, -0.05346745, 0.49517479, 0.30614935, -0.06559511, 0.96058735], dtype=np.float32)
# learning_rate = 5e-3

# case 10
# gt_name = 'gt_116_0.0006_0.0002_0.2_0.01'
# para_init = np.array([0.05674694, 0.01369266, 0.44860817, 0.22555443, 0.07814557, 0.93910208], dtype=np.float32)
# learning_rate = 5e-3

# case 11
# gt_name = 'gt_114_0.0006_-0.0002_0.2_0.01'
# para_init = np.array([-0.01287693, 0.03669609, 0.49896379, 0.17336496, -0.10310928, 0.96273004], dtype=np.float32)
# learning_rate = 1e-3

# case 12
# gt_name = 'gt_77_0.0000_-0.0004_0.2_0.01'
# para_init = np.array([-0.0113067, 0.01297076, 0.50116264, -0.02547022, -0.07791625, 0.99043562], dtype=np.float32)
# learning_rate = 1e-3

# case 13
# gt_name = 'gt_68_-0.0002_0.0002_0.2_0.01'
# para_init = np.array([0.01502009, 0.01128124, 0.45353879, -0.06081486, 0.09237081, 0.99031604], dtype=np.float32)
# learning_rate = 5e-4

# case 14
# gt_name = 'gt_60_-0.0004_0.0010_0.2_0.01'
# para_init = np.array([0.03248185, -0.02220978, 0.51864275, -0.07948681, 0.259091, 0.9444913], dtype=np.float32)
# learning_rate = 1e-3

# case 15
# gt_name = 'gt_55_-0.0004_0.0000_0.2_0.01'
# para_init = np.array([0.02040058, 0.00215043, 0.47889186, -0.06490632, 0.00800613, 0.98581098], dtype=np.float32)
# learning_rate = 1e-3

# case 15-2
# gt_name = 'gt_55_-0.0004_0.0000_0.2_0.01'
# para_init = np.array([0.0398499, 0.01571681, 0.49439321, -0.08871401, 0.00401984, 1.00536015], dtype=np.float32)
# learning_rate = 1e-3

# case 16
# gt_name = 'gt_89_0.0002_-0.0004_0.2_0.01'
# para_init = np.array([0.03202468, 0.03173961, 0.50329204, 0.04889548, -0.09832403, 0.98181468], dtype=np.float32)
# learning_rate = 5e-3

# case 17
# gt_name = 'gt_101_0.0004_-0.0004_0.2_0.01'
# para_init = np.array([0.0227518, 0.00766118, 0.47790319, 0.11375026, -0.08906584, 1.00830015], dtype=np.float32)
# learning_rate = 2e-3

# case 18
# gt_name = 'gt_104_0.0004_0.0002_0.2_0.01'
# para_init = np.array([0.03319924, 0.02002979, 0.50055042, 0.10888013, 0.03906697, 1.00946123], dtype=np.float32)
# learning_rate = 2e-3

# case 19
# gt_name = 'gt_110_0.0006_-0.0010_0.2_0.01'
# para_init = np.array([0.02557949, -0.01016392, 0.50808019, 0.16178857, -0.22469102, 0.94657539], dtype=np.float32)
# learning_rate = 2e-3

# case 20
# gt_name = 'gt_48_-0.0006_0.0010_0.2_0.01'
# para_init = np.array([0.01073053, 0.00961475, 0.4991775, -0.12749497, 0.24076442, 0.9041424], dtype=np.float32)
# learning_rate = 2e-3

# case 21
gt_name = 'gt_6_-0.0012_-0.0002_0.2_0.01'
para_init = np.array([-0.00671091, 0.01295108, 0.4984588, -0.2834613, -0.04673098, 0.93417034], dtype=np.float32)
learning_rate = 2e-3


test_idx = datetime.now().strftime("%m%d%H%M")
result_folder = f"test_imgs/results_old_{test_idx}"
rendered_imgs_folder = result_folder + '/' + f"rendered_imgs"


main(scripts_path, dataset_folder, gt_name, rendered_imgs_folder, result_folder, para_init, learning_rate=learning_rate)