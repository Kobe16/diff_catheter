'''
This file was created as an attempt to reconstruct a 3D model of a cylinder around a 
Bezier curve using PyTorch3D meshes. However, I struggled with understanding how to 
use the meshes, so I ended up ditching this idea. 
'''

import sys
from turtle import pd

sys.path.append('..')

import os
import numpy as np

# import transforms
# import bezier_interspace_transforms
sys.path.insert(1, '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts')
from bezier_set import BezierSet
import camera_settings

import torch
from pytorch3d.structures import Meshes
import pytorch3d.renderer as renderer
import pytorch3d.io as io
import pytorch3d.utils as utils

import open3d as o3d

import cv2
import matplotlib.pyplot as plt

import pdb

from construction_bezier import ConstructionBezier
from blender_catheter import BlenderRenderCatheter
from diff_render_catheter import DiffRenderCatheter
from loss_define import ContourLoss, MaskLoss

import pytorch3d

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.cm as colormap

from tqdm.notebook import tqdm


class CylinderConstruction(nn.Module): 

    def __init__(self, ): 
        ## initialize a catheter
        n_beziers = 1
        self.bezier_set = BezierSet(n_beziers)

        # self.bezier_num_samples = 101
        self.bezier_num_samples = 20
        self.bezier_surface_resolution = 50

        # self.bezier_radius = 0.0015
        self.bezier_radius = 0.5

    def createCylinderMesh(self, para_gt, p_start): 
        # Setup
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        p_mid = para_gt[0:3]
        p_end = para_gt[3:6]
        p_c2 = 4 / 3 * p_mid - 1 / 3 * p_start
        p_c1 = 4 / 3 * p_mid - 1 / 3 * p_end
        # self.control_pts = torch.vstack((p_start, c2, p_end, c1))

        sample_list = torch.linspace(0, 1, self.bezier_num_samples)

        # Get positions and normals from samples along bezier curve
        self.bezier_pos = torch.zeros(self.bezier_num_samples, 3)
        self.bezier_der = torch.zeros(self.bezier_num_samples, 3)
        self.bezier_snd_der = torch.zeros(self.bezier_num_samples, 3)
        for i, s in enumerate(sample_list):
            self.bezier_pos[i, :] = (1 - s)**3 * p_start + 3 * s * (1 - s)**2 * \
                p_c1 + 3 * (1 - s) * s**2 * p_c2 + s**3 * p_end
            self.bezier_der[i, :] = -(1 - s)**2 * p_start + ((1 - s)**2 - 2 * s *
                                                             (1 - s)) * p_c1 + (-s**2 + 2 *
                                                                                (1 - s) * s) * p_c2 + s**2 * p_end
            self.bezier_snd_der[i, :] = 6 * (1 - s) * (p_c2 - 2 * p_c1 + p_start) + 6 * s * (p_end - 2 * p_c2 + p_c1)


        print("self.bezier_pos: " + str(self.bezier_pos))

        # # Define control points for the Bézier curve
        # control_points = torch.tensor([[0.0, 0.0, 0.0],
        #                             [1.0, 0.0, 0.0],
        #                             [1.0, 1.0, 0.0],
        #                             [0.0, 1.0, 0.0]])

        # # Generate points along the Bézier curve
        # t = torch.linspace(0, 1, self.bezier_num_samples)
        # bezier_points = utils.interpolate_bezier(control_points, t)

        theta = torch.linspace(0, 2 * torch.pi, self.bezier_num_samples)
        x = self.bezier_radius * torch.cos(theta)
        y = self.bezier_radius * torch.sin(theta)
        print(theta)
        print(x)
        print(y)

        # Circle points in 3D
        circle_points = torch.stack((x, y, torch.zeros_like(x)), dim=-1) 
        print("circle_points" + str(circle_points))

        cylinder_points = self.bezier_pos.unsqueeze(1) + circle_points.unsqueeze(0)
        # Reshape to a flattened tensor
        cylinder_points = cylinder_points.view(-1, 3)  # Reshape to a flattened tensor

        num_points = self.bezier_pos.shape[0]
        num_circle_points = circle_points.shape[0]
        faces = []
        for i in range(num_points):
            for j in range(num_circle_points):
                v0 = i * num_circle_points + j
                v1 = ((i + 1) % num_points) * num_circle_points + j
                v2 = ((i + 1) % num_points) * num_circle_points + (j + 1) % num_circle_points
                v3 = i * num_circle_points + (j + 1) % num_circle_points
                faces.append([v0, v1, v2, v3])
        faces = torch.tensor(faces, dtype=torch.long)
        

        print("Faces: " + str(faces))

        # Create the mesh structure
        # mesh = Meshes(verts=[cylinder_points], faces=[faces])
        cylinder_mesh = Meshes(verts=[cylinder_points], faces=[faces])


        # Visualize the mesh using PyTorch3D's renderer or save it as an obj file
        images = renderer(cylinder_mesh)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")

        plt.show()


        # Specify the output file path
        # output_file = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/blender_imgs/cylinder_mesh_kobe.obj'

        # # Save the mesh as an OBJ file
        # io.save_obj(output_file, mesh)

if __name__ == '__main__':

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = "/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/diff_render/data"
    obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    # Load obj file
    mesh = io.load_objs_as_meshes([obj_filename], device=device)


    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    R, T = renderer.look_at_view_transform(2.7, 0, 180) 
    cameras = renderer.FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = renderer.RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = renderer.PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = renderer.MeshRenderer(
        rasterizer=renderer.MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=renderer.SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    # images = renderer(mesh)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")

    # plt.figure(figsize=(7,7))
    # texture_image=mesh.textures.maps_padded()
    # plt.imshow(texture_image.squeeze().cpu().numpy())
    # plt.axis("off")

    # plt.show()


    # para_gt = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896], dtype=torch.float)
    # p_start = torch.tensor([0.02, 0.002, 0.0])
    p_start = torch.tensor([2., 2., 0.0])


    # para_init = torch.tensor([0.02, 0.002, 0.15, 0.03, -0.05, 0.2], dtype=torch.float, requires_grad=True)
    # para_init = torch.tensor([0.01958988, 0.00195899, 0.09690406, -0.03142905, -0.0031429, 0.18200866],
    #                          dtype=torch.float)
    para_init = torch.tensor([1.958988, 0.195899, 9.690406, -3.142905, -0.31429, 18.200866],
                             dtype=torch.float)
    

    bezier_cylinder = CylinderConstruction()

    bezier_cylinder.createCylinderMesh(para_init, p_start)

