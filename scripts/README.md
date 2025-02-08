# Catheter Shape Reconstruction

The following describes the purpose of each file in the repository.

## File Descriptions

### Reconstruction

#### Main Files
- **scripts\reconstruction_single.py**
  - Sample execution script that run a single shape reconstruction process and visualize the results. 

- **scripts\catheter_reconstruction\reconst_3_loss.py**
  - File that encapsulates the core optimization algorithm with 3 loss functions, which can be invoked by the control pipeline. 

- **scripts\catheter_reconstruction\reconst_2_loss.py**
  - File that encapsulates the core optimization algorithm with 2 loss functions. 


#### Testing Files
- **scripts\test_diff_render_catheter_v2\test_contour.ipynb**
  - Contains tests for reference contour of the catheter.

- **scripts\test_diff_render_catheter_v2\test_centerline.ipynb**
  - Contains tests for reference centerline of the catheter.
  
- **scripts\test_diff_render_catheter_v2\test_read_gt&cam_proj.ipynb**
  - Contains tests for the reading of ground truth and projected centerline and contour of the catheter.

- **scripts\test_diff_render_catheter_v2\test_construction_bezier.ipynb**
  - Contains tests for the process of generating projected centerline and contour of the catheter.
  
- **scripts\test_diff_render_catheter_v2\test_process.py**
  - A comprehensive test file. For a ground truth image and a initial guess, test the reading of ground truth data, image processing, projected and reference centerline and contour, 3D visualization.

- **scripts\catheter_reconstruction\test_catheter_motion3.ipynb**
  - File used to test the accuracy of inverse solution of the motion model used in reconstruction.

- **scripts\catheter_reconstruction\test_past_frame.ipynb**
  - File used to test the accuracy of the projection of catheter in the past frames.

- **scripts\catheter_reconstruction\test_recon_cam.py**
  - Test the camera projection in the catheter shape reconstruction pipeline.

- **scripts\catheter_reconstruction\test_recons_new_v3.py**
  - This script is the new version of the catheter reconstruction algorithm. 

- **scripts\catheter_reconstruction\reconstructionOptimizer_main.py**
  - Integrated code for the new version of the catheter reconstruction algorithm, which enables large-scale testing.
  
- **scripts\test_diff_render_catheter_v2\test_recon_old2.py**
  - This script is the old version of the catheter reconstruction algorithm. It is provided for comparison and testing purposes.

- **scripts\test_diff_render_catheter_v2\recon_old_main.py**
  - Integrated code for the old version of the catheter reconstruction algorithm, which enables large-scale testing.

#### Utils
- **scripts\catheter_reconstruction\plot_3d_bezier.py**
  - This script is used for plotting 3D Bezier curves to visualize the results of the catheter reconstruction.

- **scripts\catheter_reconstruction\read_data.py**
  - This script is used for reading and visualizing the data saved during the reconstruction process.

- **scripts\bezier_set.py**
  - Script that call `blender_files\render_bezier_blender.py` to generate catheter images using Blender based on the specified catheter parameters.

- **blender_files\render_bezier_blender.py**
  - Script that call Blender to perform the rendering of the catheter images. It specifies the parameters of the rendering.

- **scripts\catheter_reconstruction\plot_optimization_loss.ipynb**
  - Plot the loss of the optimization process.

- **scripts\test_diff_render_catheter_v2\gt_generation.ipynb**
  - This Jupyter notebook is used for generating ground truth data.

### Control

#### Main Files

- **scripts\simulation_experiment.py**
  - Main pipeline of control simulation.

- **scripts\cc_catheter.py**
  - Define the class that represents the catheter in simulation.

#### Experiment Execution Files

- **scripts\experiment_execution.py**
  - Generate dataset and execute the target reaching experiment for all targets in the dataset.

- **scripts\experiment_execution_single.py**
  - Execute the target reaching experiment on designated target points in the dataset.

- **scripts\castnet_experiments.py**
  - Execute the image space target reaching experiment.

- **scripts\castnet_experiments_single.py**
  - Execute the image space target reaching experiment for sepecific target points.

- **scripts\waypoint_guidance_experiments.py**
  - Execute the waypoint tracking experiment.


#### Experiment Results Interpretation Files

- **scripts\convergence_test_interpreter.ipynb**
  - Analyze and visualize the results of the target reaching experiment.

- **scripts\result_interpreter_casnet.ipynb**
  - Visualize the result of the image space target reaching experiment.

- **scripts\result_interpreter_waypoint.py**
  - Visualize results of the waypoint tracking experiment.


#### Testing Files

- **scripts\test_data_generation.ipynb**
  - Script for testing the generation of targets of the experiment and their visualization.

- **scripts\catheter_reconstruction\cc_bezier.py**
  - Script for visualizing the conversion of constant curvature curve to bezier curve.

- **scripts\test_plot_loss.ipynb**
  - Test the method to plot control loss in the simulation pipeline (simulation_experiment.py).

- **scripts\catheter_reconstruction\test_noise_generation.py**
  - Test the noise generation method for conversion from constant curvature model to bezier curve.

- **scripts\catheter_reconstruction\catheter_workspace.py**
  - Plot the workspace of the catheter for a given set of parameters.

- **scripts\catheter_reconstruction\test_3d_to_2d.ipynb**
  - Test the camera projection method in the control simulation pipeline.

- **scripts\test_diff_render_catheter_v2\test_camera_view.py**
  - Script that test the camera projection calculation (matching between numerical computation and Blender rendering).

#### Utils

- **scripts\contour_generation.ipynb**
  - Generate contour images for waypoint guidance experiment.

- **scripts\scripts\plot_2d_shape_control.ipynb**
  - Visualize target bezier curve for convergence test of 2D shape loss.

- **scripts\target_data_generation.py**
  - Generate targets for the simulation experiments for catheter control. Extracted from scripts\experiment_execution.py.

- **scripts\target_data_modify.ipynb**
  - Modify certain data points in the dataset for target reaching experiment.

- **scripts\catheter_reconstruction\plot_3d_bezier.py**
  - Plot 3D Bezier curve (with surface). Shape reconstruction.

- **scripts\catheter_reconstruction\plot_3d_bezier_control.py**
  - Plot 3D Bezier curve (with surface). Control experiment.

- **scripts\plot_convergence_test.ipynb**
  - Plot the error curves of the target reaching experiment (version 1).

- **scripts\catheter_reconstruction\plot_control_loss_curve.ipynb**
  - Plot the error curves of the target reaching experiment (version 2).

- **scripts\catheter_reconstruction\plot_tube.py**
  - Plot 3-D schematic diagram of the Bezier curve (in paper).

- **scripts\catheter_reconstruction\plot_2d_bezier_control.ipynb**
  - Plot the schematic diagram of the 2-D control process (in paper).

- **scripts\jacobian_derivation.ipynb**
  - Script for mathematical derivation. Calculate the jacobian of T matrix.

- **scripts\test_motion_model.ipynb**
  - Script for validation of the constant curvature motion model. Validate that T @ p0 = T @ [0,0,0,1] + p0.

### Usage

- **blender_files\render_bezier_blender.py**
  - This script configures render settings, including materials, camera and lighting, and then invokes Blender to render the Bezier curve based on the specified curve parameters.

- **scripts\path_settings.py**
  - The file contains the commonly used file and directory paths for this project.

- **scripts\camera_settings.py**
  - Define camera settings (intrinsic and extrinsic parameters) for the simulation experiments. Shoule be consistent with the camera settings in Blender.

- **scripts\experiment_setup.py**
  - This file defines various experiments.


