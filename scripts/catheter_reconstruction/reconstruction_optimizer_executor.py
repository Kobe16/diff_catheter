"""
File to execute the reconstruction optimizer. This will be the main 
function of the old reconstruction script. It should include the loop
that will run the forward and backward passes n times. 
"""

import numpy as np
import torch
import cv2

from reconstruction_optimizer import ReconstructionOptimizer


class ReconstructionOptimizerExecutor(): 

    def __init__(self, p_0, para_init, n_iters, gt_images_save_path, img_save_path) -> None:

        # Have to reformat numpy arrays from column to row vectors
        self.p_0 = p_0
        self.para_init = para_init
        
        self.n_iters = n_iters
        self.gt_images_save_path = gt_images_save_path
        self.img_save_path = img_save_path

    def execute(self): 

        ###========================================================
        ### 1) SET TO GPU OR CPU COMPUTING
        ###========================================================
        if torch.cuda.is_available():
            gpu_or_cpu = torch.device("cuda:0")
            torch.cuda.set_device(gpu_or_cpu)
        else:
            gpu_or_cpu = torch.device("cpu")

        catheter_optimize_model = ReconstructionOptimizer(self.p_0, self.para_init, img_ref_binary, gpu_or_cpu).to(gpu_or_cpu)
