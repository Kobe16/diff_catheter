# Catheter Shape Reconstruction

The following describes the purpose of each file in the repository.

## File Descriptions

### Ground Truth Generation
- **scripts/test_diff_render_catheter_v2/gt_generation.ipynb**
  - This Jupyter notebook is used for generating ground truth data.

### Reconstruction

#### Reconstruction Scripts
- **scripts/catheter_reconstruction/test_recons_new_v3.py**
  - This script is the new version of the catheter reconstruction algorithm. 

- **scripts\catheter_reconstruction\reconstructionOptimizer_main.py**
  - Integrated code for the new version of the catheter reconstruction algorithm, which enables large-scale testing.
  
- **scripts/test_diff_render_catheter_v2/test_recon_old2.py**
  - This script is the old version of the catheter reconstruction algorithm. It is provided for comparison and testing purposes.

- **scripts\test_diff_render_catheter_v2\recon_old_main.py**
  - Integrated code for the old version of the catheter reconstruction algorithm, which enables large-scale testing.

#### Testing Files
- **scripts/test_diff_render_catheter_v2/test_contour.ipynb**
  - Contains tests for reference contour of the catheter.

- **scripts/test_diff_render_catheter_v2/test_centerline.ipynb**
  - Contains tests for reference centerline of the catheter.
  
- **scripts/test_diff_render_catheter_v2/test_read_gt&cam_proj.ipynb**
  - Contains tests for the reading of ground truth and projected centerline and contour of the catheter.

- **scripts\test_diff_render_catheter_v2\test_construction_bezier.ipynb**
  - Contains tests for the process of generating projected centerline and contour of the catheter.
  
- **scripts\test_diff_render_catheter_v2\test_process.py**
  - A comprehensive test file. For a ground truth image and a initial guess, test the reading of ground truth data, image processing, projected and reference centerline and contour, 3D visualization.

- **scripts\catheter_reconstruction\test_catheter_motion3.ipynb**
  - File used to test the accuracy of inverse solution of the motion model used in reconstruction.

- **scripts\catheter_reconstruction\test_past_frame.ipynb**
  - File used to test the accuracy of the projection of catheter in the past frames.

#### Utils
- **scripts/catheter_reconstruction/plot_3d_bezier.py**
  - This script is used for plotting 3D Bezier curves to visualize the results of the catheter reconstruction.

- **scripts\catheter_reconstruction\read_data.py**
  - This script is used for reading and visualizing the data saved during the reconstruction process.

- **scripts\bezier_set.py**
  - Script that call `blender_files\render_bezier_blender.py` to generate catheter images using Blender based on the specified catheter parameters.

- **blender_files\render_bezier_blender.py**
  - Script that call Blender to perform the rendering of the catheter images. It specifies the parameters of the rendering.

### Control

#### Math
- **scripts\jacobian_derivation.ipynb**
  - Script for mathematical derivation. Calculate the jacobian of T matrix.

- **scripts\test_motion_model.ipynb**
  - Script for validation of the constant curvature motion model. Validate that T @ p0 = T @ [0,0,0,1] + p0.

#### Testing Files
- **scripts\test_data_generation.ipynb**
  - Script for testing the generation of targets of the experiment and their visualization.

- **scripts\catheter_reconstruction\cc_bezier.py**
  - Script for visualizing the conversion of constant curvature curve to bezier curve.

- **scripts\test_plot_loss.ipynb**
  - Test the method to plot control loss in the simulation pipeline (simulation_experiment.py).


## Usage

Each script and notebook has been designed for specific tasks within the project. Detailed instructions and usage examples are included within the respective files.

## License

[Specify your project's license here]

## Contact

For any questions or further information, please contact [Your Contact Information].

