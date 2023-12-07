# ARCLab-CCCatheter

## Dependencies
- Blender       (enter in scripts/path_settings.py the path to the directory containing Blender executable)
- torch         (pip)
   - (1.13.0 for Apple Silicon)
- opencv-python (pip)
- matplotlib    (pip)
- numpy         (pip)
- shapely       (pip)

- scikit-image  (pip)
- torchvision   (pip)
   - (0.14.0 for Apple Silicon)
- pytorch3d     (pip)
   - (install from github for Apple Silicon)

## Routine of Operation
1. Modify scripts/path_settings.py
2. Add directory /target_parameters under /data
3. Setup the experiment executors and run
4. Setup the experiment interpretors in the same way as the experiment executors and run

If you want Blender to render curves in the background without interrupting script execution, 
add the '--background' tag to subprocess.run([]) inside scripts/bezier_set.py

## Full Path Tree
```bash
ARCLab-CCCatheter
│
├──blender_files
│  ├──render_bezier_blender.py          ## Blender script used in main pipeline
│  ├──render_bezier.blend               ## (Optional) for visualization
│  └──render_bezier.blend1              ## (Optional) goes with .blend
├──data
│  ├──bezier_specs                      ## temporary storage to be used during execution of main pipeline
│  ├──contour_images                    ## data of contour images
│  ├──rendered_images                   ## for visualizing rendered images
│  ├──rendered_videos                   ## for visualizing rendered videos
│  └──target_parameters                 ## data generated for convergence experiment
├──results                              ## results of experiments and tables and figures produced by result interpreters
└──scripts
   ├──reconstruction_scripts            ## Fei's reconstruction algorithms
   │  ├──reconst_sim_opt2pts.py
   │  └──reconst_sim_opt3pts.py 
   ├──bezier_interspace_transforms.py   ## calculations for interspace transforms 
   ├──bezier_set.py                     ## Calls Blender script to render Bezier curves
   ├──camera_settings.py
   ├──castnet_experiments.py            ## executor for heatmap experiment
   ├──cc_catheter.py                    ## basic catheter class
   ├──contour_tracer.py                 ## trace contour for shape on an image
   ├──convert_camera_settings.py
   ├──data_generation.py                ## data generator for convergence experiment
   ├──experiment_execution.py           ## executor for convergence experiment
   ├──experiment_setup.py               ## parameter settings for all methods
   ├──identifier_conversions.py         ## conversions between method names, identifiers, and indices
   ├──path_settings.py
   ├──result_interpreter_castnet.py     ## result interpreter for heatmap experiment
   ├──result_interpreter_general.py     ## result interpreter for convergence experiment
   ├──result_interpreter_waypoint.py    ## result interpreter for waypoint experiment
   ├──simulation_experiment.py          ## wrap basic catheter class in a pipeline
   ├──temp_image_modifier.py            ## (tangent)
   ├──transforms.py                     ## calculations for unispace transforms (these are also used by interspace transforms)
   ├──waypoint_guidance_experiments.py  ## executor for waypoint experiment
   └──write_video_from_img.py           ## (tangent) 
```

## Updated Full Path Tree (11/21/23)
```bash 
├── README.md
├── blender_files
│   ├── render_bezier.blend
│   ├── render_bezier.blend1
│   ├── render_bezier_blender.py                         ## Blender parser to generate inputted info (i.e., curve)
│   ├── rod_original (copy).blend
│   ├── rod_original (copy).blend1
│   ├── rod_original.blend
│   └── rod_original.blend1
├── file_structure.txt
├── results
│   └── table_1_p3d.csv
└── scripts
    ├── __pycache__
    │   ├── bezier_set.cpython-310.pyc
    │   ├── bezier_set.cpython-38.pyc
    │   ├── camera_settings.cpython-310.pyc
    │   ├── camera_settings.cpython-38.pyc
    │   ├── path_settings.cpython-310.pyc
    │   └── path_settings.cpython-38.pyc
    ├── bezier_interspace_transforms.py
    ├── bezier_set.py                                    ## Takes in bezier curve info and renders it in Blender
    ├── cam_test.py
    ├── camera_settings.py
    ├── castnet_experiments.py
    ├── cc_catheter.py
    ├── contour_tracer.py
    ├── convert_camera_settings.py
    ├── data_generation.py
    ├── diff_render
    │   ├── BKups
    │   │   └── diff_render_2pts_BKUP.py
    │   ├── __pycache__
    │   │   ├── bezier_set.cpython-310.pyc
    │   │   ├── blender_catheter.cpython-310.pyc
    │   │   ├── blender_catheter.cpython-38.pyc
    │   │   ├── build_diff_model.cpython-38.pyc
    │   │   ├── construction_bezier.cpython-310.pyc
    │   │   ├── construction_bezier.cpython-38.pyc
    │   │   ├── diff_render_catheter.cpython-310.pyc
    │   │   ├── diff_render_catheter.cpython-38.pyc
    │   │   ├── loss_define.cpython-310.pyc
    │   │   └── loss_define.cpython-38.pyc
    │   ├── blender_catheter.py
    │   ├── blender_imgs
    │   │   ├── cylinder_primitve.mtl
    │   │   ├── cylinder_primitve.obj
    │   │   ├── diff_render_1.mtl
    │   │   ├── diff_render_1.npy
    │   │   ├── diff_render_1.obj
    │   │   ├── diff_render_1.png
    │   │   ├── diff_render_2.npy
    │   │   └── diff_render_2.png
    │   ├── build_diff_model.py
    │   ├── camera_position_optimization_with_differentiable_rendering.ipynb
    │   ├── construction_bezier.py
    │   ├── cylinder.mat
    │   ├── cylinder_primitive.csv
    │   ├── cylinder_primitive.npy
    │   ├── diff_open_blender.py
    │   ├── diff_optimize_2pts.py
    │   ├── diff_optimize_2pts_ok.py
    │   ├── diff_render_catheter.py
    │   ├── get-pip.py
    │   ├── install_torch3d.py
    │   ├── loss_define.py
    │   ├── projectCurve.ipynb
    │   ├── test_code.m
    │   ├── test_cyl_constr.py
    │   ├── test_diff_render.ipynb
    │   ├── test_generate_primitive.ipynb
    │   ├── test_torch3d_rendering.ipynb
    │   └── test_torch3d_rendering_CLEAN.ipynb
    ├── experiment_execution.py
    ├── experiment_setup.py
    ├── hough_bezier.py
    ├── identifier_conversions.py
    ├── path_settings.py
    ├── postprocessing.py
    ├── real_robot_experiment.py
    ├── real_robot_experiment_executor.py
    ├── reconstruction_scripts
    │   ├── 05_DataAssociation-Clutter.py
    │   ├── 07_PDATutorial.py
    │   ├── PDA_test.py
    │   ├── reconst_sim_opt2pts.py
    │   ├── reconst_sim_opt2pts_PDA.py
    │   ├── reconst_sim_opt3pts.py
    │   └── usage.md
    ├── result_interpreter_castnet.py
    ├── result_interpreter_general.py
    ├── result_interpreter_waypoint.py
    ├── simulation_experiment.py
    ├── temp_image_modifier.py
    ├── test_diff_render_catheter               ## OG attempt at reconst                      
    │   └── ...
    ├── test_diff_render_catheter_v2            ## SRC Reconst code directory
    │   ├── blender_imgs
    │   │   ├── test_catheter_gt1.npy           ## Sample npy of blender catheter
    │   │   └── test_catheter_gt1.png           ## Sample png of blender catheter
    │   ├── important_imgs
    │   │   ├── render_59_tipBP-mean.jpg
    │   │   ├── render_59_tipBP-sum.jpg
    │   │   └── render_59_tiponly-mean.jpg
    │   ├── math_calcuations.py                 ## Extra file for math calculations
    │   ├── test_blender_catheter.py            ## Render bezier catheter in blender. Calls upon `scripts/bezier_set.py`
    │   ├── test_diff_render_catheter_v2.py     ## Unused
    │   ├── test_graph_random.py                ## Script to test and graph inter-pipeline plots/images
    │   ├── test_loss_define_v2.py              ## Loss functions for optimization algorithm
    │   ├── test_optimize_v2.py                 ## Main script for catheter reconstruction optimization
    │   ├── test_plot_2_curves.py               ## Same as `test_reconst_v2.py`, but can plot 2 curves in 3d space
    │   └── test_reconst_v2.py                  ## Script to generate 3d bezier catheter model & get 2d projection
    ├── transforms.py
    ├── waypoint_guidance_experiments.py
    └── write_video_from_img.py

```

## Logical Hierarchy
```bash
## Upper Level
├──result_interpreter_general.py
│  ├──identifier_conversions.py
│  └──experiment_execution.py
│     ├──data_generation.py
│     ├──experiment_setup.py
│     └──simulation_experiment.py
│
├──result_interpreter_castnet.py
│  ├──identifier_conversions.py
│  └──castnet_experiments.py
│     ├──experiment_setup.py
│     └──simulation_experiment.py
│  
└──result_interpreter_waypoint.py
   ├──identifier_conversions.py
   └──waypoint_guidance_experiments.py
      ├──contour_tracer.py
      ├──experiment_setup.py
      └──simulation_experiment.py

## Lower Level
simulation_experiment.py
├──reconst_sim_opt2pts.py
└──cc_catheter.py
   ├──transforms.py                  
   ├──bezier_interspace_transforms.py
   └──bezier_set.py  
      └──render_bezier_blender.py
```
