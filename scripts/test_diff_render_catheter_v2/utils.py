import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image(img_save_path):
    img_ref_rgb = cv2.imread(img_save_path)
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)
    (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 2, 255, cv2.THRESH_BINARY)
    img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)
    
    return img_ref_binary

def read_gt_params(cc_specs_path):
    """
    Ground truth: [p_start, p_end, c1, c2]^T, (1, 4, 3) matrix (numpy array)
    return: [p1, p_end], length 6 vector
    Conversion from 4 bezier control points to 3 bezier control points
    """
    
    para_gt_np = np.load(cc_specs_path)
    matrix = np.squeeze(para_gt_np)
    c1 = matrix[2]
    c2 = matrix[3]
    p_start = matrix[0]
    p_end = matrix[1]
    p_mid = 3/4 * (c1 + p_end/3)
    p1 = 2*p_mid - 0.5*p_start - 0.5*p_end
    result_vector = np.concatenate((p1, p_end))
    return result_vector

def bezier_curve_3d(control_points, num_points=100):
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 3))

        for i in range(num_points):
            curve[i] = (1 - t[i]) ** 2 * control_points[0] + \
                    2 * (1 - t[i]) * t[i] * control_points[1] + \
                    t[i] ** 2 * control_points[2]

        return curve
    
# def plot_3D_bezier_curve(control_points, control_points_gt, control_points_init, save_path=None):
#     # Generate the Bezier curve
#     curve = bezier_curve_3d(control_points)
#     curve_gt = bezier_curve_3d(control_points_gt)
#     curve_init = bezier_curve_3d(control_points_init)
    
    
#     # Plotting the Bezier curve
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--')
#     ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Optimized Result')

#     ax.plot(control_points_gt[:, 0], control_points_gt[:, 1], control_points_gt[:, 2], 'bo--')
#     ax.plot(curve_gt[:, 0], curve_gt[:, 1], curve_gt[:, 2], 'b-', label='Ground Truth')

#     ax.plot(control_points_init[:, 0], control_points_init[:, 1], control_points_init[:, 2], 'go--')
#     ax.plot(curve_init[:, 0], curve_init[:, 1], curve_init[:, 2], 'g-', label='Initial Guess')

#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Optimization Result')
#     ax.legend()
    
#     if save_path is not None:
#         if not os.path.exists(os.path.dirname(save_path)):
#             os.makedirs(os.path.dirname(save_path))
#         plt.savefig(save_path)

#     plt.show()
    
def plot_3D_bezier_curve(control_points=None, control_points_gt=None, control_points_init=None, save_path=None, equal=False):
    """
    All bezier control points input are in standard format.
    [p_start, p1, p_end]^T, 3x3 matrix (numpy array)
    """
    # Plotting the Bezier curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if control_points is not None:
        curve = bezier_curve_3d(control_points)
        ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], 'ro--')
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'r-', label='Optimized Result')

    if control_points_gt is not None:
        curve_gt = bezier_curve_3d(control_points_gt)
        ax.plot(control_points_gt[:, 0], control_points_gt[:, 1], control_points_gt[:, 2], 'bo--')
        ax.plot(curve_gt[:, 0], curve_gt[:, 1], curve_gt[:, 2], 'b-', label='Ground Truth')

    if control_points_init is not None:
        curve_init = bezier_curve_3d(control_points_init)
        ax.plot(control_points_init[:, 0], control_points_init[:, 1], control_points_init[:, 2], 'go--')
        ax.plot(curve_init[:, 0], curve_init[:, 1], curve_init[:, 2], 'g-', label='Initial Guess')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimization Result')
    if equal:
        ax.set_aspect('equal')
    ax.legend()
    
    # Set axis ranges
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([0, 1])
    
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)

    plt.show()
    
def plot_2d_end_effector_loss(proj_end_effector_loss_history, full_path=None):
    # Given array of values proj_end_effector_loss_history, create plot of loss vs. iterations
    iterations_x_axis_proj = list(range(len(proj_end_effector_loss_history)))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig1.suptitle('2D Tip Euclidean Distance Loss History')
    ax1.plot(iterations_x_axis_proj, proj_end_effector_loss_history)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Euclidean Distance Loss (Pixels)')
    ax1.set_xlim([0, len(proj_end_effector_loss_history)])
    # ax1.set_ylim([0, 80])
    ax1.set_ylim(bottom=0)
    ax1.grid(True)
    
    if full_path is not None:
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        plt.savefig(full_path)

    plt.show()
    
def plot_3d_end_effector_loss(d3d_end_effector_loss_history, full_path=None):
    iterations_x_axis_3d = list(range(len(d3d_end_effector_loss_history)))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('3D Tip Euclidean Distance Loss History')
    ax2.plot(iterations_x_axis_3d, d3d_end_effector_loss_history)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Euclidean Distance Loss (m)')
    ax2.set_xlim([0, len(d3d_end_effector_loss_history)])
    ax2.set_ylim(bottom=0)
    # ax2.set_ylim([0, 0.05])
    ax2.grid(True)

    if full_path is not None:
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        plt.savefig(full_path)

    plt.show()
    
def plot_3d_mid_control_point_loss(d3d_mid_control_point_loss_history, full_path=None):
    iterations_x_axis_3d = list(range(len(d3d_mid_control_point_loss_history)))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('3D Shape (Middle Control Point) Loss History')
    ax2.plot(iterations_x_axis_3d, d3d_mid_control_point_loss_history)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Euclidean Distance Loss (m)')
    ax2.set_xlim([0, len(d3d_mid_control_point_loss_history)])
    ax2.set_ylim(bottom=0)
    # ax2.set_ylim([0, 0.05])
    ax2.grid(True)

    if full_path is not None:
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        plt.savefig(full_path)

    plt.show()
    
def plot_total_loss(loss_history, full_path=None, log_scale=False):
    iterations_x_axis_loss = list(range(len(loss_history)))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('Total Loss History')
    ax2.plot(iterations_x_axis_loss, loss_history)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Total Loss')
    ax2.set_xlim([0, len(loss_history)])
    ax2.set_ylim(bottom=0)
    if log_scale:
        ax2.set_ylim(bottom=1)
        ax2.set_yscale('log')
        ax2.set_ylabel('Total Loss (log scale)')
    ax2.grid(True)

    if full_path is not None:
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        plt.savefig(full_path)

    plt.show()
    
def plot_contour_loss(loss_contour_history, full_path=None, log_scale=False):
    iterations_x_axis_loss_contour = list(range(len(loss_contour_history)))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('Contour Loss History')
    ax2.plot(iterations_x_axis_loss_contour, loss_contour_history)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Contour Loss')
    ax2.set_xlim([0, len(loss_contour_history)])
    ax2.set_ylim(bottom=0)
    if log_scale:
        ax2.set_ylim(bottom=1)
        ax2.set_yscale('log')
        ax2.set_ylabel('Contour Loss (log scale)')
    ax2.grid(True)

    if full_path is not None:
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        plt.savefig(full_path)

    plt.show()
    
def plot_tip_loss(loss_tip_distance_history, full_path=None, log_scale=False):
    iterations_x_axis_loss_tip = list(range(len(loss_tip_distance_history)))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('Tip Loss History')
    ax2.plot(iterations_x_axis_loss_tip, loss_tip_distance_history)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Tip Loss')
    ax2.set_xlim([0, len(loss_tip_distance_history)])
    ax2.set_ylim(bottom=0)
    if log_scale:
        ax2.set_ylim(bottom=0.0001)
        ax2.set_yscale('log')
        ax2.set_ylabel('Tip Loss (log scale)')
    ax2.grid(True)

    if full_path is not None:
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        plt.savefig(full_path)

    plt.show()
    
def plot_motion_model_loss(loss_motion_model_history, full_path=None):
    iterations_x_axis_loss_mm = list(range(len(loss_motion_model_history)))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle('Motion Model Loss History')
    ax2.plot(iterations_x_axis_loss_mm, loss_motion_model_history)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Motion Model Loss')
    ax2.set_xlim([0, len(loss_motion_model_history)])
    ax2.set_ylim(bottom=0)
    # ax2.set_ylim([0, 0.05])
    ax2.grid(True)

    if full_path is not None:
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        plt.savefig(full_path)

    plt.show()
    
def bezier_conversion_4_to_3(bezier_params_4):
    """
    Input: [p_start, p_end, c1, c2]^T, (4, 3) matrix (numpy array)
    Output: [p_start, p1, p_end]^T, (3, 3) matrix (numpy array)
    """
    c1 = bezier_params_4[2]
    c2 = bezier_params_4[3]
    p_start = bezier_params_4[0]
    p_end = bezier_params_4[1]
    
    p_mid = 3/4 * (c1 + p_end/3)
    p1 = 2*p_mid - 0.5*p_start - 0.5*p_end
    
    bezier_params_3 = np.vstack((p_start, p1, p_end))
    
    return bezier_params_3
    