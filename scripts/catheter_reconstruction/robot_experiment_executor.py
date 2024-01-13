"""
File to input starting parameters and execute the entire experiment. 
Will have a robot_experiment object that it will call to execute. 
"""
import os
import sys
import numpy as np

from robot_experiment import RobotExperiment
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import path_settings
from experiment_setup import experiments




### Universal parameters
p_0 = np.array([2e-2, 2e-3, 0])
r = 0.01
n_iter = 10
noise_percentage = 0.25
ux_init = 0.00001
uy_init = 0.00001
l_init = 0.2
n_control_iters = 3
ux_controls = [0.00002, 0.0001, 0.0005]
uy_controls = [0.00002, 0.0001, 0.0005]


# identifiers_of_interest = ['UN012', 'UN013', 'IA012', 'IA013', 'IA112', 'IA113']

identifier = 'UN002'
data_alias = 'RR01'


## set_targets
x_target = 50
y_target = 100


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Running real experiment experiment for ', identifier)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

exp = experiments[identifier]
dof = exp['dof']
loss_2d = exp['loss_2d']
tip_loss = exp['tip_loss']
use_reconstruction = exp['use_reconstruction']

# if not use_reconstruction:
#     print('[ERROR] Real robot experiments requires reconstruction')
#     exit()

interspace = exp['interspace']
viewpoint_mode = exp['viewpoint_mode']
damping_weights = exp['damping_weights']
n_mid_points = exp['n_mid_points']

render_mode = 2

method_dir = os.path.join(path_settings.results_dir, identifier)
if not os.path.isdir(method_dir):
    os.mkdir(method_dir)

data_dir_outer = os.path.join(method_dir, data_alias)
if not os.path.isdir(data_dir_outer):
    os.mkdir(data_dir_outer)

data_dir = os.path.join(data_dir_outer, str(x_target).zfill(4) + '_' + str(y_target).zfill(4))
images_save_dir = os.path.join(data_dir, 'images')
gt_images_save_dir = os.path.join(data_dir, 'gt_images')

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
    os.mkdir(images_save_dir)

ux = ux_init
uy = uy_init
l = l_init

sim_exp = RobotExperiment(dof, loss_2d, tip_loss, interspace, viewpoint_mode, damping_weights, noise_percentage, n_iter, render_mode, n_control_iters, ux_controls, uy_controls)
sim_exp.set_paths(images_save_dir, gt_images_save_dir)
sim_exp.set_general_parameters(p_0, r, n_mid_points, l)
sim_exp.set_2dof_parameters(ux, uy, x_target, y_target)
sim_exp.execute()

