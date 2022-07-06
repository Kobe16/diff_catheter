import numpy as np

import pytorch3d.utils as torch3d_utils
import pytorch3d.renderer as torch3d_render
import pytorch3d.io as torch3d_io
import pytorch3d.structures as torch3d_structures

import torch

import matplotlib.pyplot as plt

import pdb


class DiffRenderCatheter:

    def __init__(self, camera_extrinsics, camera_intrinsics, gpu_or_cpu):

        # ## Set the cuda device
        # if torch.cuda.is_available():
        #     self.gpu_or_cpu = torch.device("cuda:0")
        #     torch.cuda.set_device(self.gpu_or_cpu)
        # else:
        #     self.gpu_or_cpu = torch.device("cpu")

        self.gpu_or_cpu = gpu_or_cpu

        self.IMG_WIDTH = 480
        self.IMG_HEIGHT = 640

        self.setRenderingCamera(camera_extrinsics, camera_intrinsics)

    def loadCylinderPrimitive(self, path):
        # Load the obj and ignore the textures and materials.
        self.verts, self.faces_idx, _ = torch3d_io.load_obj(path)

        self.faces = self.faces_idx.verts_idx

        # Initialize each vertex to be white in color.textures
        verts_rgb = torch.zeros_like(self.verts.float()).unsqueeze(0)  # (1, N, 3)
        self.textures = torch3d_render.TexturesVertex(verts_features=verts_rgb.to(self.gpu_or_cpu))

        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        self.cylinder_primitive_mesh = torch3d_structures.Meshes(verts=[self.verts.to(self.gpu_or_cpu)],
                                                                 faces=[self.faces.to(self.gpu_or_cpu)],
                                                                 textures=self.textures)

    def updateCylinderPrimitive(self, updated_verts):

        cylinder_updated_verts = updated_verts.to(self.gpu_or_cpu)
        cylinder_primitive_verts = self.cylinder_primitive_mesh.verts_list()[0]
        cylinder_deformed_verts = cylinder_updated_verts - cylinder_primitive_verts

        # pdb.set_trace()

        self.updated_cylinder_primitive_mesh = self.cylinder_primitive_mesh.offset_verts(
            cylinder_deformed_verts.float())

    def setRenderingCamera(self, camera_extrinsics, camera_intrinsics):
        self.cam_RT_H = torch.as_tensor(camera_extrinsics, device=self.gpu_or_cpu, dtype=torch.float)
        self.cam_K = torch.as_tensor(camera_intrinsics, device=self.gpu_or_cpu, dtype=torch.float)

        rot = (self.cam_RT_H[0:3, 0:3]).unsqueeze(0)
        tvec = (self.cam_RT_H[0:3, 3]).unsqueeze(0)
        camK = self.cam_K.unsqueeze(0)
        image_size = torch.as_tensor([[self.IMG_WIDTH, self.IMG_HEIGHT]], device=self.gpu_or_cpu)

        self.render_cameras = torch3d_utils.cameras_from_opencv_projection(R=rot,
                                                                           tvec=tvec,
                                                                           camera_matrix=camK,
                                                                           image_size=image_size)

        # focal = torch.tensor((cam_K[0, 0],cam_K[1, 1]), dtype=torch.float32).unsqueeze(0)
        # principle = torch.tensor((cam_K[0, 2],cam_K[1, 2]), dtype=torch.float32).unsqueeze(0)
        # cameras = PerspectiveCameras(device=device, R=rot, T=trans, focal_length=focal, principal_point=principle, image_size=((480, 640),))

    def renderDeformedMesh(self, save_img_path):

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # WIDTH x HEIGHT. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        raster_settings = torch3d_render.RasterizationSettings(
            image_size=(self.IMG_WIDTH, self.IMG_HEIGHT),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
        # -z direction.
        self.lights = torch3d_render.PointLights(device=self.gpu_or_cpu, location=[[0.0, 0.0, -3.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        self.renderer_catheter = torch3d_render.MeshRenderer(
            rasterizer=torch3d_render.MeshRasterizer(cameras=self.render_cameras, raster_settings=raster_settings),
            shader=torch3d_render.SoftPhongShader(device=self.gpu_or_cpu,
                                                  cameras=self.render_cameras,
                                                  lights=self.lights))

        self.render_catheter_img = self.renderer_catheter(self.updated_cylinder_primitive_mesh)
        # self.render_catheter_img = self.renderer_catheter(self.cylinder_primitive_mesh)

        fig = plt.figure(figsize=(7, 5))
        plt.imshow(self.render_catheter_img[0, ..., 0:3].cpu().detach().numpy())
        fig.tight_layout()
        # plt.axis("off")
        # plt.show()
        fig.savefig(save_img_path)
        plt.close(fig)
