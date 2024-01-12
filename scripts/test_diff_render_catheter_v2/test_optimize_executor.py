import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from test_reconst_v2 import ConstructionBezier
from test_optimize_v2 import CatheterOptimizeModel
from test_loss_define_v2 import ChamferLossWholeImage, ContourChamferLoss, \
    TipChamferLoss, BoundaryPointChamferLoss, TipDistanceLoss, BoundaryPointDistanceLoss, \
    GenerateRefData



class ReconstructionOptimizeScriptExecutor(): 

    def __init__(self, p_0, para_init, n_optimize_iters, img_get_path, img_save_path) -> None:

        # Have to reformat numpy arrays from column to row vectors
        self.p_0 = p_0
        self.para_init = para_init
        
        self.n_optimize_iters = n_optimize_iters
        self.img_get_path = img_get_path
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

        ###========================================================
        ### 2) VARIABLES FOR BEZIER CURVE CONSTRUCTION
        ###========================================================
        # p_start used for SRC presentation
        # p_start = torch.tensor([0.02, 0.008, 0.054])

        # case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/test_diff_render_catheter_v2/blender_imgs/test_catheter_gt1'
        img_get_path_full = self.img_get_path + '.png'

        '''
        Create binary mask of catheter: 
            1) Grayscale the ref img, 
            2) threshold the grayscaled img, 
            3) Creates a binary image by replacing all 
                pixel values equal to 255 with 1 (leaves
                other pixel values unchanged)
        '''
        img_ref_rgb = cv2.imread(img_get_path_full)
        img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)
        (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)


        ###========================================================
        ### 3) SET UP AND RUN OPTIMIZATION MODEL
        ###========================================================
        catheter_optimize_model = CatheterOptimizeModel(self.p_0, self.para_init, img_ref_binary, gpu_or_cpu).to(gpu_or_cpu)

        print("Model Parameters:")
        for name, param in catheter_optimize_model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        optimizer = torch.optim.Adam(catheter_optimize_model.parameters(), lr=1e-2)

        # Run the optimization loop
        loop = tqdm(range(100))
        for loop_id in loop:
            print("\n================================================================================================================")
            print("loop_id: ", loop_id)
            
            img_save_path_full = self.img_save_path + 'render_' + str(loop_id) + '.jpg'

            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()

            # Run the forward pass
            loss = catheter_optimize_model(img_save_path_full)

            # Print gradients for all parameters before backward pass
            print("Gradients BEFORE BACKWARD PASS:")
            for name, param in catheter_optimize_model.named_parameters():
                if param.grad is not None:
                    print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")  # Print the norm of the gradient
                else:
                    print(f"{name}: No gradient computed")

            # Run the backward pass
            loss.backward(retain_graph=True)

            # Print gradients for all parameters after backward pass
            print("Gradients AFTER BACKWARD PASS:")
            for name, param in catheter_optimize_model.named_parameters():
                if param.grad is not None:
                    print(f"Parameter: {name}, Gradient: {param.grad.norm().item()}")
                else:
                    print(f"{name}: No gradient computed")

            # Update the parameters
            optimizer.step()


            # Print and inspect the updated parameters
            for name, param in catheter_optimize_model.named_parameters():
                print(f"Parameter: {name}, Updated Value: {param.data}")

            # Update the progress bar
            loop.set_description('Optimizing')

            # Update the loss
            loop.set_postfix(loss=loss.item())

            print("Loss: ", loss.item())

