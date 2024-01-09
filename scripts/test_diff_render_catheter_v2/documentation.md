Explanation of Fei's old code: 

Start at `real_robot_experiment.py`: 
1) Starts off by setting all the parameters, such as catheter start point, radius, length, DoF, etc. 
2) It uses `cc_catheter.py` to represent the catheter object. This object stores the catheter information and it also performs all of the calculations for the catheter control. 
3) The goal of `real_robot_experiment.py` is to control the catheter through inverse kinematics such that the catheter goes to an inputted target position. It is basically a big inverse kinematics solver that will run for n iterations to move the catheter so that it reaches a target position. 
4) After setting up parameters, the main body of `real_robot_experiment.py` runs: 
  1) The `cc_catheter.py` runs the method `update_xdof_params`, which finds the update in parameters that brings the catheter 'closer' to the target position. It uses inverse kinematics, specifcially the damped least squares method. 
  2) `cc_catheter.py` runs the method `calculate_cc_points`, which calculates n-# of midpoints along the constant curvaturve curve using the catheter's homoegeneous transformation matrix. You decide the number of points n along the curve that you want. These points are stored in `self.cc_pt_list` or `self.target_cc_pt_list`. 
  3) `cc_catheter.py` runs the method `convert_cc_points_to_2d`, which takes the cc-points you just calculated, and converts them to the 2D image plane using image projection. 
  4) `cc_catheter.py` runs the method `calculate_beziers_control_points`, which converts the the cc-points you just calculated into cc points // 2 number of bezier curves. Sets these curves into `self.bezier_set`. Note, the `self.bezier_set` object is built from the BezierSet class. This class can directly work with blender to create curves. 
  5) `cc_catheter.py` runs the method `render_beziers`, which renders the bezier curves (that we just set in `self.bezier_set`) to blender. 
  6) Run fei's reconstruction?????
