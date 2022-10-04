import os
import sys
import numpy as np
import imageio
import argparse
import scipy.misc
import matplotlib.pyplot
import imageio
from PIL import Image

num = 0

case_name = 'loss_mask_4k'
# case_name = 'loss_mask_end'
# case_name = 'loss_mask_only'

##### =========================================================
##### =========================================================
st_x = 1
ed_x = 201
save_path = '/media/fei/DATA_Fei/Datasets/Octupus_Arm/octupus_data_F/icra2023_videos/'
src_dir = '/home/fei/icra2023_diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/' + case_name + '/version_video_selected/frame_117/'

writer = imageio.get_writer(save_path + case_name + '_frame_117_step.mp4', fps=10)
# writer = imageio.get_writer(save_path + case_name + '_frame_105_surface.mp4', fps=10)
for i in range(st_x, ed_x, 1):

    filename = os.path.join(src_dir, 'video_step_' + '%d.png' % i)
    # filename = os.path.join(src_dir, '3d_surface_step_' + '%d.png' % i)
    print(filename)
    img = imageio.imread(filename)

    #images.append(img)
    writer.append_data(img)
    num = num + 1

print("total frame num: ", num)
writer.close()

##### =========================================================
##### =========================================================

# st_x = 0
# ed_x = 32
# save_path = '/media/fei/DATA_Fei/Datasets/Octupus_Arm/octupus_data_F/icra2023_videos/'
# src_dir = '/home/fei/icra2023_diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/' + case_name + '/version_video_allframe_3d_surface/'

# writer = imageio.get_writer(save_path + case_name + '_2d_render_finalstep_allframe.mp4', fps=2)
# # writer = imageio.get_writer(save_path + case_name + '_3d_surface_finalstep_allframe.mp4', fps=2)
# for i in range(st_x, ed_x, 1):

#     frame_id_list = (93, 94, 95, 97, 98, 104, 105, 108, 109, 110, 111, 112, \
#                         113, 114, 115, 116, 117, 118, 119, 120, \
#                         121, 122, 123, 124, 125, 135, 136, 137, \
#                         138, 139, 140, 147)

#     frame_id = frame_id_list[i]

#     filename = os.path.join(src_dir, '2d_render_finalstep_frame_' + '%d.png' % frame_id)
#     # filename = os.path.join(src_dir, '3d_surface_finalstep_frame_' + '%d.png' % frame_id)
#     print(filename)
#     img = imageio.imread(filename)

#     #images.append(img)
#     writer.append_data(img)
#     num = num + 1

# print("total frame num: ", num)
# writer.close()
