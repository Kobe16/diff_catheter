import sys

sys.path.append('..')

import os
import numpy as np

# import transforms
# import bezier_interspace_transforms
from bezier_set import BezierSet
import camera_settings

import torch

import open3d as o3d


class DiffRenderCatheter:
    def __init__(self):

        ## initialize camera parameters
        self.setCameraParams(camera_settings.a, camera_settings.b, camera_settings.center_x, camera_settings.center_y,
                             camera_settings.image_size_x, camera_settings.image_size_y, camera_settings.extrinsics,
                             camera_settings.intrinsics)

        ## initialize a catheter
        n_beziers = 1
        self.bezier_set = BezierSet(n_beziers)

        mesh_cylinder = o3d.geometry.TriangleMesh.create_mesh_cylinder(radius=1.0, height=2.0, resolution=20, split=4)

    def setCameraParams(self, fx, fy, cx, cy, size_x, size_y, camera_extrinsics, camera_intrinsics):
        """
        Set intrinsic and extrinsic camera parameters

        Args:
            fx (float): horizontal direction focal length
            fy (float): vertical direction focal length
            cx (float): horizontal center of image
            cy (float): vertical center of image
            size_x (int): width of image
            size_y (int): height of image
            camera_extrinsics ((4, 4) numpy array): RT matrix 
            camera_intrinsics ((3, 3) numpy array): K matrix 
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.size_x = size_x
        self.size_y = size_y
        self.cam_RT_H = camera_extrinsics
        self.cam_K = camera_intrinsics

        # camera E parameters
        cam_RT_H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        invert_y = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.cam_RT_H = torch.matmul(invert_y, cam_RT_H)

    def getBezierCurve(self, para_gt, p_start):

        p_mid = para_gt[0:3]
        p_end = para_gt[3:6]
        p_c1 = 4 / 3 * p_mid - 1 / 3 * p_start
        p_c2 = 4 / 3 * p_mid - 1 / 3 * p_end
        # self.control_pts = torch.vstack((p_start, c2, p_end, c1))

        self.num_samples = 100
        sample_list = torch.linspace(0, 1, self.num_samples)

        # Get positions and normals from samples along bezier curve
        self.bezier_pos = torch.zeros(self.num_samples, 3)
        self.bezier_der = torch.zeros(self.num_samples, 3)
        self.bezier_snd_der = torch.zeros(self.num_samples, 3)
        for i, s in enumerate(sample_list):
            self.bezier_pos[i, :] = (1 - s)**3 * p_start + 3 * s * (1 - s)**2 * \
                p_c1 + 3 * (1 - s) * s**2 * p_c2 + s**3 * p_end
            self.bezier_der[i, :] = -(1 - s)**2 * p_start + ((1 - s)**2 - 2 * s *
                                                             (1 - s)) * p_c1 + (-s**2 + 2 *
                                                                                (1 - s) * s) * p_c2 + s**2 * p_end
            self.bezier_snd_der[i, :] = 6 * (1 - s) * (p_c2 - 2 * p_c1 + p_start) + 6 * s * (p_end - 2 * p_c2 + p_c1)

        # Convert positions and normals to camera frame
        pos_bezier_H = torch.cat((self.bezier_pos, torch.ones(self.num_samples, 1)), dim=1)

        pos_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(pos_bezier_H, 0, 1)), 0, 1)
        self.pos_bezier_cam = pos_bezier_cam_H[1:, :-1]

        der_bezier_H = torch.cat((self.bezier_der, torch.zeros((self.num_samples, 1))), dim=1)
        der_bezier_cam_H = torch.transpose(torch.matmul(self.cam_RT_H, torch.transpose(der_bezier_H[1:, :], 0, 1)), 0,
                                           1)
        self.der_bezier_cam = der_bezier_cam_H[:, :-1]

    def getBezierTNB(self, bezier_pos, bezier_der, bezier_snd_der):

        bezier_tangent = bezier_der / torch.linalg.norm(bezier_der, ord=2, dim=1)


class BlenderRenderCatheter:
    def __init__(self):

        ## initialize a catheter
        n_beziers = 1
        self.bezier_set = BezierSet(n_beziers)

    def set_bezier_in_blender(self, para_gt, p_start):
        p_mid = para_gt[0:3]
        p_end = para_gt[3:6]

        # c = (p_mid - (p_start / 4) - (p_end / 4)) * 2
        c1 = 4 / 3 * p_mid - 1 / 3 * p_end
        c2 = 4 / 3 * p_mid - 1 / 3 * p_start

        self.bezier_set.enter_spec(p_start, p_end, c1, c2)

    def render_bezier_in_blender(self,
                                 curve_specs_path,
                                 img_save_path,
                                 target_specs_path=None,
                                 viewpoint_mode=1,
                                 transparent_mode=0):
        """
        Render Bezier curves according to the curve specs.

        Args:
            curve_specs_path (path string to npy file): curve specs is a (n, 3) numpy array where each
                row specifies the start point, middle control point, and end point of a Bezier curve
            img_save_path (path string to png file): path to save rendered image
            target_specs_path (path string to npy file): target specs is a (n, 3) numpy array where each
                row specifies the 3D position of a target point. If this path is set to None,
                target points will be not rendered
            viewpoint_mode (1 or 2): camera view of rendered image, 1 for endoscopic view, 2 for side view
            transparent_mode (0 or 1): whether to make the background transparent for the rendered image,
                0 for not transparent, 1 for transparent
        """
        if not self.bezier_set:
            print('[ERROR] [CCCatheter] self.bezier_set invalid. Run calculate_beziers_control_points() first')
            exit()

        self.bezier_set.print_specs()
        self.bezier_set.write_specs(curve_specs_path)
        self.bezier_set.render(img_save_path, target_specs_path, viewpoint_mode, transparent_mode)


if __name__ == '__main__':

    blender_catheter = BlenderRenderCatheter()

    para_gt = torch.tensor([0.02003904, 0.0016096, 0.10205799, 0.02489567, -0.04695673, 0.196168896], dtype=torch.float)
    p_start = torch.tensor([0.02, 0.002, 0.0])

    case_naming = '/home/fei/ARCLab-CCCatheter/scripts/diff_render/blender_imgs/diff_render_1'
    img_save_path = case_naming + '.png'
    cc_specs_path = case_naming + '.npy'
    target_specs_path = None
    viewpoint_mode = 1
    transparent_mode = 0

    blender_catheter.set_bezier_in_blender(para_gt.detach().numpy(), p_start.detach().numpy())

    blender_catheter.render_bezier_in_blender(cc_specs_path, img_save_path, target_specs_path, viewpoint_mode,
                                              transparent_mode)
