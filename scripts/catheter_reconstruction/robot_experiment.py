"""
File to execute the main parts of the robot experiment. 
It will create ground truth images, loop through each image and perform 
reconstruction optimizer on it (It calls the reconstruction_optimizer_executor 
object to execute the reconstruction script). 
"""
from reconstruction_optimizer_executor import ReconstructionOptimizerExecutor


class RobotExperiment(): 

    def __init__(self, dof, interspace, viewpoint_mode, damping_weights, noise_percentage, n_iter, render_mode, ux_controls=None, uy_controls=None): 
        """
        Args:
            dof (1, 2, or 3): DoF of control (1 DoF is not fully implemented currently)
            interspace (0, 1, or 2): interspace of control, 0 for unispace, 1 for Bezier interspace
                with (theta, phi) parameterization, 2 for Bezier interspace with (ux, uy) parameterization
            viewpoint_mode (1 or 2): camera view of rendered image, 1 for endoscopic view, 2 for side view
            damping_weights (list of 3 floats): n-th term corresponds to the damping weight of the n-th DoF control feedback
            noise_percentage: gaussian noise will be applied to the feedback. 
                The variance of that noise would be noise_percentage * feedback
            n_iter (int): number of total iteration of reconstruction optimization 
            render_mode (0, 1, or 2): 0 for rendering no image, 1 for only rendering the image after
                the last iteration, 2 for rendering all image 
            ux_controls (float array): list of changes to actuation 'ux' (for 2 DoF and 3 DoF)
            uy_controls (float array): list of changes to actuation 'uy' (for 2 DoF and 3 DoF)
        """
        self.dof = dof
        self.interspace = interspace
        self.viewpoint_mode = viewpoint_mode
        self.damping_weights = damping_weights
        self.noise_percentage = noise_percentage
        self.n_iter = n_iter
        self.render_mode = render_mode
        self.ux_controls = ux_controls
        self.uy_controls = uy_controls

    def set_paths(self, images_save_dir, gt_images_save_dir): 
        """
        Args:
            images_save_dir (path string to directory): directory to save rendered images
            gt_images_save_dir (path string to directory): directory to save ground truth images
        """
        self.images_save_dir = images_save_dir
        self.gt_images_save_dir = gt_images_save_dir

    def set_general_parameters(self, p_0, r, n_mid_points, l):
        """
        Args:
            p_0 ((3,) numpy array): start point of catheter
            r (float): cross section radius of catheter
            n_mid_points (int): number of middle control points (likely just 1 mid point)
            l (float): length of catheter
        """
        self.p_0 = p_0
        self.r = r
        self.n_mid_points = n_mid_points
        self.l = l

    def set_1dof_parameters(self, u, phi):
        """
        Set parameters for 1DoF (not fully implemented currently)

        Args:
            u (float): tendon length (responsible for catheter bending)
            phi (radians as float): phi parameter
        """
        self.u = u
        self.phi = phi

    def set_2dof_parameters(self, ux, uy):
        """
        Set parameters for 2DoF

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
        """
        self.ux = ux
        self.uy = uy

    def set_3dof_parameters(self, ux, uy, l):
        """
        Set parameters for 3DoF (not fully implemented currently)

        Args:
            ux (float): 1st pair of tendon length (responsible for catheter bending)
            uy (float): 2nd pair of tendon length (responsible for catheter bending)
        """
        self.ux = ux
        self.uy = uy
        self.l = l

    def get_gt_images(self): 
        """
        Get ground truth Blender-generated images of catheter using ux_controls and uy_controls. 
        """


    def execute(self): 
        """
        Runs reconstruction script on each frame of the ground truth pictures. 
        """

        optimizer_executor = ReconstructionOptimizerExecutor()


