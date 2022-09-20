clc
clear

addpath('./npy-matlab/npy-matlab') 

close all


mask_4k_final_img = '/home/fei/diff_catheter/scripts/diff_render_octupus/torch3d_rendered_imgs/real_dataset_render/loss_mask_4k/'


frame_id = 93
render_3d_centerline_path = strcat(mask_4k_final_img, 'frame_', int2str(frame_id), '/render_3d_centerline_frame_', int2str(frame_id), '.npy') 
render_3d_centerline = readNPY(render_3d_centerline_path);

figure(1)
plot3(render_3d_centerline(:,1), render_3d_centerline(:,2), render_3d_centerline(:,3))

frame_id = 105
render_3d_centerline_path = strcat(mask_4k_final_img, 'frame_', int2str(frame_id), '/render_3d_centerline_frame_', int2str(frame_id), '.npy') 
render_3d_centerline = readNPY(render_3d_centerline_path);
hold on
plot3(render_3d_centerline(:,1), render_3d_centerline(:,2), render_3d_centerline(:,3))


frame_id = 123
render_3d_centerline_path = strcat(mask_4k_final_img, 'frame_', int2str(frame_id), '/render_3d_centerline_frame_', int2str(frame_id), '.npy') 
render_3d_centerline = readNPY(render_3d_centerline_path);
hold on
plot3(render_3d_centerline(:,1), render_3d_centerline(:,2), render_3d_centerline(:,3))

frame_id = 136
render_3d_centerline_path = strcat(mask_4k_final_img, 'frame_', int2str(frame_id), '/render_3d_centerline_frame_', int2str(frame_id), '.npy') 
render_3d_centerline = readNPY(render_3d_centerline_path);
hold on
plot3(render_3d_centerline(:,1), render_3d_centerline(:,2), render_3d_centerline(:,3))

grid on
% daspect([1,1,1])