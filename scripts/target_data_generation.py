"""
Generate targets for the simulation experiments for catheter control.
Extracted from scripts/experiment_execution.py
"""

import os
import numpy as np
import camera_settings
import path_settings
from data_generation import DataGeneration


# Universal parameters
p_0 = np.array([2e-2, 2e-3, 0])
r = 0.01
n_iter = 10 
n_data = 100
l_init = 0.2

data_alias = 'D' + str(0).zfill(2)
data_save_path = os.path.join(path_settings.target_parameters_dir, data_alias + '.npy')
s_list = [0.5, 1]

data_gen = DataGeneration(n_data, p_0, r, l_init, s_list, data_save_path)
data_gen.set_target_ranges(0.0005, 0.01, 0.0005, 0.01, 0.2, 0.2) # (-0.005, 0.005, -0.005, 0.005, 0.1, 0.5)
data_gen.set_camera_params(camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y, camera_settings.image_size_x, camera_settings.image_size_y, camera_settings.extrinsics)
data_gen.generate_data()
target_parameters = np.load(data_save_path)

data_gen.visualize_targets(path_settings.target_parameters_dir, 1, n_iter)



