import numpy as np
from utils import *

scripts_path = 'E:/OneDrive - UC San Diego/UCSD/Lab/Catheter/diff_catheter/scripts/test_diff_render_catheter_v2'
result_folder = "test_imgs/results_complete_07151405"
filename = "bezier_params_1800.npy"
full_path = scripts_path + '/' + result_folder + '/' + filename

control_points = np.load(full_path)

dataset_folder = "gt_dataset4"
gt_name = 'gt_35_-0.0008_0.0008_0.2_0.01'
case_naming = scripts_path + '/' + dataset_folder + '/' + gt_name
img_save_path = case_naming + '.png'
cc_specs_path = case_naming + '.npy'

para_gt_np_4 = np.load(cc_specs_path)
para_gt_np_4 = np.squeeze(para_gt_np_4)
para_gt_np_3 = bezier_conversion_4_to_3(para_gt_np_4)
print("Ground truth parameters: ", para_gt_np_3)

para_init = np.array([[0.02, 0.002, 0],
                   [0.02, -0.01, 0.436],
                   [-0.2, -0.17, 0.73]], dtype=np.float32)

plot_3D_bezier_curve(control_points, control_points_gt=para_gt_np_3, control_points_init=para_init)
