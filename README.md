# ARCLab-CCCatheter

## Dependencies
- Blender       (enter in scripts/path_settings.py the path to the directory containing Blender executable)
- torch         (pip)
- opencv-python (pip)
- matplotlib    (pip)
- numpy         (pip)
- shapely       (pip)

- scikit-image  (pip)

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
