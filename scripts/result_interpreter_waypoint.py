""" Visualize results of the waypoint tracking experiment """

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

import path_settings
import contour_tracer



n_iter = 5
identifiers_of_interest = ['EXP009', 'EXP010']
image_names = ['rectangle'] # heart, rectangle


### Table 5
table_5_mean = np.zeros((4, 12))
table_5_std = np.zeros((4, 12))

for i, image_name in enumerate(image_names):

    image_path = os.path.join(path_settings.contour_images_dir, image_name + '.png')

    ## Get waypoints from image contour
    ct = contour_tracer.ContourTracer(image_path)
    waypoints_2d = ct.trace_contour()
    print('Number of waypoints: ', waypoints_2d.shape[0])

    ## Sample waypoints
    selected_indices = np.arange(0, waypoints_2d.shape[0], 50)
    waypoints_2d_selected = waypoints_2d[selected_indices, :]

    n_data = waypoints_2d_selected.shape[0]
    print('Number of sampled waypoints: ', n_data)

    for j, identifier in enumerate(identifiers_of_interest):
        method_dir = os.path.join(path_settings.results_dir, identifier)
        data_dir_outer = os.path.join(method_dir, image_name)

        p2d_loss_norms = []

        for k in range(n_data):

            data_dir = os.path.join(data_dir_outer, str(k).zfill(4))

            p2d_report_path = os.path.join(data_dir, 'p2d_poses.npy')
            p2d_report = np.load(p2d_report_path)

            p2d_loss = p2d_report[-1, -1, :] - p2d_report[-2, -1, :]
            p2d_loss_norm = np.linalg.norm(p2d_loss, 2)
            p2d_loss_norms.append(p2d_loss_norm)

        p2d_loss_norms = np.array(p2d_loss_norms)
        table_5_mean[i, j] = np.mean(p2d_loss_norms)
        table_5_std[i, j] = np.std(p2d_loss_norms)
    
np.savetxt(os.path.join(path_settings.results_dir, 'table_5_mean.csv'), table_5_mean, delimiter=',', fmt='%f')
np.savetxt(os.path.join(path_settings.results_dir, 'table_5_std.csv'), table_5_std, delimiter=',', fmt='%f')
print("Mean: ", table_5_mean)
print("Standard Deviation: ", table_5_std)



### Figure set 5

for i, image_name in enumerate(image_names):

    image_path = os.path.join(path_settings.contour_images_dir, image_name + '.png')

    ## Get waypoints from image contour
    ct = contour_tracer.ContourTracer(image_path)
    waypoints_2d = ct.trace_contour()
    print('Number of waypoints: ', waypoints_2d.shape[0])

    ## Draw contour and resized images
    contour_image_path = image_path[:-4] + '_contour.png'
    resized_image_path = image_path[:-4] + '_resized.png'

    ## Sample waypoints
    selected_indices = np.arange(0, waypoints_2d.shape[0], 50)
    waypoints_2d_selected = waypoints_2d[selected_indices, :]

    n_data = waypoints_2d_selected.shape[0]
    print('Number of sampled waypoints: ', n_data)

    ## Draw waypoints on specified image
    #if i == len(image_names) - 1:
    #    resized_image_path = '/home/inffzy/Desktop/ARCLab/ARCLab-CCCatheter/data/contour_images/tumor4595_resized.png' 

    resized_image = cv2.imread(resized_image_path)

    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    overwrite = True
    
    for idx, identifier in enumerate(identifiers_of_interest):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        method_dir = os.path.join(path_settings.results_dir, identifier)
        data_dir_outer = os.path.join(method_dir, image_name)

        x_2d_old = None
        y_2d_old = None
        
        if overwrite:
            resized_image_temp = resized_image
            if idx == 0:
                color_2d = (50, 50, 220)  # Red, (0, 0, 255)
            else:
                color_2d = (0, 140, 255)  # Orange, (0, 165, 255)
        else:
            resized_image_temp = cv2.imread(resized_image_path)
            color_2d = (50, 50, 220)  # Red

        for i in range(n_data):

            data_dir = os.path.join(data_dir_outer, str(i).zfill(4))
            
            p3d_report_path = os.path.join(data_dir, 'p3d_poses.npy')
            p2d_report_path = os.path.join(data_dir, 'p2d_poses.npy')
            p3d_report = np.load(p3d_report_path)
            p2d_report = np.load(p2d_report_path)

            #rendered_image_path = os.path.join(data_dir, 'images', str(n_iter).zfill(3) + '.png')
            #rendered_image = cv2.imread(rendered_image_path)
            #resized_image_temp = cv2.addWeighted(resized_image_temp, 0.9, rendered_image, 0.1, 0)
            
            def rectangle(cx, cy, size):
                half = size // 2  # Half the size for calculations
                pts = np.array([
                    [cx - half, cy - half],  # Top-left
                    [cx + half, cy - half],  # Top-right
                    [cx + half, cy + half],  # Bottom-right
                    [cx - half, cy + half]   # Bottom-left
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                return pts

            ## Plot 2D point
            x_2d = int(p2d_report[-2, -1, 0])
            y_2d = int(p2d_report[-2, -1, 1])
            
            if i == 0:
                pts = rectangle(x_2d, y_2d, 12)
                resized_image_temp = cv2.polylines(resized_image_temp, [pts], isClosed=True, color=color_2d, thickness=2)
                # resized_image_temp = cv2.fillPoly(resized_image_temp, pts, color=color_2d)
            else:    
                resized_image_temp = cv2.circle(resized_image_temp, (x_2d, y_2d), radius=3, color=color_2d, thickness=2)

            if i > 0:
                x_start = (x_2d + 2 * x_2d_old) // 3
                y_start = (y_2d + 2 * y_2d_old) // 3
                x_end = (2 * x_2d + x_2d_old) // 3
                y_end = (2 * y_2d + y_2d_old) // 3

                resized_image_temp = cv2.arrowedLine(resized_image_temp, (x_start, y_start), (x_end, y_end), color=color_2d, thickness=2, tipLength=0.3)
            
            x_2d_old = x_2d
            y_2d_old = y_2d              

            ## Plot 3D point
            x_3d = p3d_report[-2, -1, 0]
            y_3d = p3d_report[-2, -1, 1]
            z_3d = p3d_report[-2, -1, 2]
            ax.scatter(x_3d, y_3d, z_3d, marker='o')

        cv2.imwrite(os.path.join(path_settings.results_dir, identifier + '_' + image_name + '_p2d.png'), resized_image_temp)
        print('Saved image: ', os.path.join(path_settings.results_dir, identifier + '_' + image_name + '_p2d.png'))
        #plt.show()
        #pickle.dump(fig_3d, open(os.path.join(path_settings.results_dir, identifier + '_' + image_name + '_p3d.pickle'), 'wb'))

        #figx = pickle.load(open(os.path.join(path_settings.results_dir, identifier + '_' + image_name + '_p3d.pickle'), 'rb'))
        #plt.show()