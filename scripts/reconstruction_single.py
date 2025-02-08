"Sample execution script that run a single shape reconstruction process and visualize the results."

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from datetime import datetime

from catheter_reconstruction.utils import *
from catheter_reconstruction.reconst_3_loss import *

def random_deviation(point, min_distance=0.01, max_distance=0.05):
    # Generate a random direction as a unit vector
    random_direction = np.random.randn(3)
    random_direction /= np.linalg.norm(random_direction)
    # Generate a random length between min_distance and max_distance
    random_length = np.random.uniform(min_distance, max_distance)
    # Calculate the offset
    offset = random_direction * random_length
    # Return the point after applying the offset
    return point + offset, random_length

folder_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/results/test_diff_render_catheter_v2/gt_dataset6/'
gt_name = 'gt_18_0.0006_0.0006_0.2_0.01'
gt_img_path = folder_path + gt_name + '.png'
gt_specs_path = folder_path + gt_name + '.npy'

p_0 = np.array([2e-2, 2e-3, 0])
para_gt_np = read_gt_params(gt_specs_path)
para_init_np = para_gt_np.copy()
para_init_np[:3], deviation1 = random_deviation(para_gt_np[:3])
para_init_np[-3:], deviation2 = random_deviation(para_gt_np[-3:])

print("Initialized parameters: ", para_init_np)
print("Deviation of middle control point: ", deviation1)
print("Deviation of end point: ", deviation2)



iteration = 0
l = 0.2
para_init = torch.tensor(para_init_np, dtype=torch.float32)
gt_img_path_list = ["gt_17_0.0006_0.0004_0.2_0.01.png"]
gt_img_path_list = [folder_path + img_name for img_name in gt_img_path_list]
delta_u_list = [[0, 0.0002]]

test_idx = datetime.now().strftime("%m%d%H%M")
result_save_path = folder_path + f"figs/results_{test_idx}"

result, loss = optimize_3_loss(gt_img_path, gt_specs_path, iteration, result_save_path, para_init, gt_img_path_list, delta_u_list, l, learning_rate = 1e-2, max_iterations = 1000, convergence_threshold=1e-3)