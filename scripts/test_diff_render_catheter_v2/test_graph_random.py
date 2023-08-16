
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from test_reconst_v2 import ConstructionBezier
from test_loss_define_v2 import ChamferLossWholeImage, ContourChamferLoss, TipChamferLoss

if __name__ == "__main__": 

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
    # Parameters to plot: 
    p_start = torch.tensor([0.02, 0.008, 0.054])
    # para_final = torch.tensor([0.02, 0.002, 0.1000, 0.0096, -0.0080,  0.1969, -0.0414, -0.0131,  0.2820], dtype=torch.float, requires_grad=False)

    para_final = torch.tensor([0.0365, 0.0036,  0.1202,  0.0056, -0.0166, 0.1645], dtype=torch.float, requires_grad=False)

    case_naming = '/Users/kobeyang/Downloads/Programming/ECESRIP/diff_catheter/scripts/test_diff_render_catheter_v2/blender_imgs/test_catheter_gt1'
    img_save_path = case_naming + '.png'

    img_ref_rgb = cv2.imread(img_save_path)
    img_ref_gray = cv2.cvtColor(img_ref_rgb, cv2.COLOR_BGR2GRAY)
    (thresh, img_ref_thresh) = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_ref_binary = np.where(img_ref_thresh == 255, 1, img_ref_thresh)
    image_ref = torch.from_numpy(img_ref_binary.astype(np.float32))


    tip_chamfer_loss = TipChamferLoss(device=gpu_or_cpu)
    tip_chamfer_loss.to(gpu_or_cpu)


    ###========================================================
    ### 3) SETTING UP BEZIER CURVE CONSTRUCTION
    ###========================================================
    build_bezier = ConstructionBezier()
    build_bezier.loadRawImage(img_save_path)


    ###========================================================
    ### 4) RUNNING BEZIER CURVE CONSTRUCTION
    ###========================================================
    # Generate the Bezier curve cylinder mesh points
    build_bezier.getBezierCurveCylinder(p_start, para_final)

    # Plot 3D Bezier Cylinder mesh points
    build_bezier.plot3dBezierCylinder()

    # Get 2D projected Bezier Cylinder mesh points
    build_bezier.getCylinderMeshProjImg()

    # Get 2D projected Bezier Centerline points (tip and boundary points)
    build_bezier.getBezierProjImg()

    # Plot 2D projected Bezier Cylinder mesh points and tip/boundary points
    build_bezier.draw2DCylinderImage()

    # Plot ALL 2d projected points
    build_bezier.plotAll2dProjPoints()

    ###========================================================
    ### 4) PLOT 2D CENTERLINE FROM REFERENCE IMAGE
    ###========================================================

    # Get 2d center line from reference image (using skeletonization)
    tip_chamfer_loss.get_raw_centerline(image_ref)
    centerline_ref = tip_chamfer_loss.img_raw_skeleton
    # print("centerline_ref shape: ", centerline_ref.shape)
    # print("centerline_ref: ", centerline_ref)
    
    # Plot the points in centerline_ref 
    fig1, ax1 = plt.subplots()

    '''
    centerline_ref.shape: (# of points, 2)
    centerline_ref[:, 1]: x coordinates (width)
    centerline_ref[:, 0]: y coordinates (height)
    '''

    ax1.plot(centerline_ref[:, 1], centerline_ref[:, 0])
    ax1.set_title('centerline_ref')
    ax1.set_xlim([0, 640])
    ax1.set_ylim([480, 0])
    plt.show()


    og_plot_points = torch.tensor([[0.02, 0.002, 0.0], 
                                [0.01958988, 0.00195899, 0.09690406], 
                                [-0.03142905, -0.0031429, 0.18200866]])

    cylinder_mesh_points = torch.tensor([[[ 1.9779e-02,  3.0329e-03,  2.8545e-05],
            [ 1.9741e-02,  1.6434e-03,  7.1235e-05],
            [ 2.0473e-02,  2.6711e-03, -1.3091e-04],
            [ 1.9129e-02,  2.5521e-03,  1.9757e-04],
            [ 2.0585e-02,  2.8103e-03, -1.6135e-04],
            [ 2.0749e-02,  1.5594e-03, -1.7081e-04],
            [ 1.9914e-02,  1.7948e-03,  2.5864e-05],
            [ 2.0771e-02,  2.1720e-03, -1.9098e-04],
            [ 1.9507e-02,  3.1482e-03,  9.1569e-05],
            [ 2.0176e-02,  1.4250e-03, -2.8660e-05],
            [ 1.8906e-02,  2.5228e-03,  2.5223e-04],
            [ 1.9290e-02,  2.6043e-03,  1.5721e-04],
            [ 2.0856e-02,  2.4042e-03, -2.1713e-04],
            [ 1.8654e-02,  1.5339e-03,  3.3731e-04],
            [ 2.0483e-02,  1.5850e-03, -1.0686e-04],
            [ 1.9237e-02,  2.2687e-03,  1.7830e-04],
            [ 2.0138e-02,  1.3864e-03, -1.8491e-05],
            [ 1.9460e-02,  1.8301e-03,  1.3496e-04],
            [ 1.9373e-02,  2.2309e-03,  1.4633e-04],
            [ 1.9933e-02,  3.2737e-03, -1.4518e-05]],

            [[ 2.3688e-02,  1.2760e-03,  1.7019e-02],
            [ 2.2547e-02,  2.0164e-03,  1.7214e-02],
            [ 2.3651e-02,  1.8043e-03,  1.7016e-02],
            [ 2.3137e-02,  3.6252e-03,  1.7077e-02],
            [ 2.4400e-02,  2.3501e-03,  1.6869e-02],
            [ 2.2291e-02,  2.2542e-03,  1.7257e-02],
            [ 2.4688e-02,  2.9876e-03,  1.6804e-02],
            [ 2.3384e-02,  1.0139e-03,  1.7079e-02],
            [ 2.2836e-02,  1.5304e-03,  1.7170e-02],
            [ 2.3650e-02,  1.7753e-03,  1.7017e-02],
            [ 2.3879e-02,  3.6124e-03,  1.6941e-02],
            [ 2.3953e-02,  2.9485e-03,  1.6940e-02],
            [ 2.4864e-02,  1.9740e-03,  1.6791e-02],
            [ 2.4340e-02,  3.1566e-03,  1.6865e-02],
            [ 2.2294e-02,  2.3987e-03,  1.7254e-02],
            [ 2.3871e-02,  1.9208e-03,  1.6974e-02],
            [ 2.2506e-02,  2.8960e-03,  1.7206e-02],
            [ 2.2246e-02,  2.7512e-03,  1.7256e-02],
            [ 2.2625e-02,  1.7976e-03,  1.7204e-02],
            [ 2.3307e-02,  1.3024e-03,  1.7088e-02]],

            [[ 2.5923e-02,  2.6878e-03,  3.3808e-02],
            [ 2.4801e-02,  2.4691e-03,  3.3948e-02],
            [ 2.7233e-02,  2.1369e-03,  3.3654e-02],
            [ 2.6412e-02,  1.6754e-03,  3.3761e-02],
            [ 2.5956e-02,  1.3937e-03,  3.3820e-02],
            [ 2.6044e-02,  3.8503e-03,  3.3779e-02],
            [ 2.6591e-02,  3.9473e-03,  3.3711e-02],
            [ 2.6151e-02,  2.5187e-03,  3.3782e-02],
            [ 2.7126e-02,  3.5103e-03,  3.3651e-02],
            [ 2.6122e-02,  2.1264e-03,  3.3791e-02],
            [ 2.5306e-02,  3.7996e-03,  3.3870e-02],
            [ 2.6799e-02,  2.1109e-03,  3.3708e-02],
            [ 2.6418e-02,  2.6287e-03,  3.3748e-02],
            [ 2.5832e-02,  2.2346e-03,  3.3825e-02],
            [ 2.7050e-02,  3.1758e-03,  3.3664e-02],
            [ 2.5879e-02,  3.3720e-03,  3.3805e-02],
            [ 2.5543e-02,  2.7456e-03,  3.3854e-02],
            [ 2.5056e-02,  1.8515e-03,  3.3925e-02],
            [ 2.7075e-02,  2.8380e-03,  3.3665e-02],
            [ 2.6937e-02,  3.4674e-03,  3.3674e-02]],

            [[ 2.8900e-02,  3.1254e-03,  5.0222e-02],
            [ 2.7691e-02,  3.5292e-03,  5.0292e-02],
            [ 2.7662e-02,  2.5601e-03,  5.0299e-02],
            [ 2.7831e-02,  1.8609e-03,  5.0293e-02],
            [ 2.7032e-02,  2.2418e-03,  5.0339e-02],
            [ 2.8376e-02,  3.4261e-03,  5.0251e-02],
            [ 2.8326e-02,  2.7316e-03,  5.0258e-02],
            [ 2.9055e-02,  2.4624e-03,  5.0216e-02],
            [ 2.8760e-02,  2.5867e-03,  5.0233e-02],
            [ 2.7119e-02,  3.7759e-03,  5.0325e-02],
            [ 2.6963e-02,  4.0519e-03,  5.0332e-02],
            [ 2.9169e-02,  2.5816e-03,  5.0209e-02],
            [ 2.7763e-02,  2.4376e-03,  5.0294e-02],
            [ 2.9071e-02,  2.8789e-03,  5.0213e-02],
            [ 2.7198e-02,  2.2133e-03,  5.0329e-02],
            [ 2.7480e-02,  2.6209e-03,  5.0310e-02],
            [ 2.7467e-02,  3.6874e-03,  5.0304e-02],
            [ 2.6418e-02,  2.4942e-03,  5.0375e-02],
            [ 2.6746e-02,  1.8171e-03,  5.0359e-02],
            [ 2.8066e-02,  1.9014e-03,  5.0279e-02]],

            [[ 2.8990e-02,  2.3449e-03,  6.6573e-02],
            [ 2.7868e-02,  2.6447e-03,  6.6568e-02],
            [ 2.9126e-02,  2.9524e-03,  6.6573e-02],
            [ 2.8384e-02,  3.5675e-03,  6.6571e-02],
            [ 2.8495e-02,  2.7785e-03,  6.6571e-02],
            [ 2.8713e-02,  1.4707e-03,  6.6571e-02],
            [ 2.8640e-02,  4.1988e-03,  6.6572e-02],
            [ 2.7998e-02,  3.1042e-03,  6.6569e-02],
            [ 2.8426e-02,  1.8542e-03,  6.6570e-02],
            [ 2.7075e-02,  1.8967e-03,  6.6564e-02],
            [ 2.9108e-02,  2.2251e-03,  6.6573e-02],
            [ 2.8939e-02,  3.9668e-03,  6.6573e-02],
            [ 2.8618e-02,  4.0232e-03,  6.6572e-02],
            [ 2.8944e-02,  2.3243e-03,  6.6572e-02],
            [ 2.8059e-02,  3.2492e-03,  6.6569e-02],
            [ 2.7270e-02,  3.9222e-03,  6.6566e-02],
            [ 2.8721e-02,  2.9529e-03,  6.6572e-02],
            [ 2.8127e-02,  2.2410e-03,  6.6569e-02],
            [ 2.8427e-02,  3.4479e-03,  6.6571e-02],
            [ 2.8835e-02,  2.8371e-03,  6.6572e-02]],

            [[ 2.6997e-02,  3.6775e-03,  8.2563e-02],
            [ 2.7322e-02,  2.2934e-03,  8.2577e-02],
            [ 2.7110e-02,  3.2695e-03,  8.2568e-02],
            [ 2.6783e-02,  3.0811e-03,  8.2544e-02],
            [ 2.8651e-02,  3.3469e-03,  8.2678e-02],
            [ 2.8480e-02,  2.2301e-03,  8.2658e-02],
            [ 2.8481e-02,  2.1047e-03,  8.2657e-02],
            [ 2.8491e-02,  3.2635e-03,  8.2666e-02],
            [ 2.7578e-02,  2.2617e-03,  8.2594e-02],
            [ 2.6471e-02,  1.8473e-03,  8.2513e-02],
            [ 2.6549e-02,  2.6896e-03,  8.2525e-02],
            [ 2.8778e-02,  2.9883e-03,  8.2684e-02],
            [ 2.7533e-02,  3.1426e-03,  8.2597e-02],
            [ 2.7965e-02,  2.3515e-03,  8.2622e-02],
            [ 2.8371e-02,  2.4473e-03,  8.2652e-02],
            [ 2.6608e-02,  3.2214e-03,  8.2533e-02],
            [ 2.6473e-02,  3.0111e-03,  8.2522e-02],
            [ 2.7149e-02,  1.4767e-03,  8.2559e-02],
            [ 2.6900e-02,  1.6687e-03,  8.2542e-02],
            [ 2.7983e-02,  3.4441e-03,  8.2631e-02]],

            [[ 2.4977e-02,  3.3664e-03,  9.8259e-02],
            [ 2.6776e-02,  2.6820e-03,  9.8500e-02],
            [ 2.4650e-02,  2.3296e-03,  9.8199e-02],
            [ 2.4814e-02,  1.7575e-03,  9.8214e-02],
            [ 2.6876e-02,  2.1412e-03,  9.8506e-02],
            [ 2.6545e-02,  2.9503e-03,  9.8471e-02],
            [ 2.6887e-02,  1.5439e-03,  9.8499e-02],
            [ 2.6243e-02,  2.0008e-03,  9.8416e-02],
            [ 2.5921e-02,  2.6912e-03,  9.8381e-02],
            [ 2.5604e-02,  1.8957e-03,  9.8326e-02],
            [ 2.5690e-02,  3.2439e-03,  9.8357e-02],
            [ 2.4866e-02,  3.3661e-03,  9.8244e-02],
            [ 2.6631e-02,  1.5842e-03,  9.8464e-02],
            [ 2.6376e-02,  3.3619e-03,  9.8454e-02],
            [ 2.6735e-02,  3.7585e-03,  9.8509e-02],
            [ 2.6063e-02,  1.8360e-03,  9.8389e-02],
            [ 2.6090e-02,  2.7653e-03,  9.8406e-02],
            [ 2.5287e-02,  2.2928e-03,  9.8287e-02],
            [ 2.5397e-02,  1.2373e-03,  9.8288e-02],
            [ 2.6450e-02,  3.4311e-03,  9.8465e-02]],

            [[ 2.2761e-02,  1.3492e-03,  1.1380e-01],
            [ 2.2821e-02,  2.5599e-03,  1.1384e-01],
            [ 2.2196e-02,  2.8059e-03,  1.1371e-01],
            [ 2.3650e-02,  3.0592e-03,  1.1402e-01],
            [ 2.4123e-02,  2.2819e-03,  1.1411e-01],
            [ 2.3468e-02,  2.9291e-03,  1.1398e-01],
            [ 2.4402e-02,  2.3610e-03,  1.1417e-01],
            [ 2.2973e-02,  1.4210e-03,  1.1385e-01],
            [ 2.2378e-02,  1.2404e-03,  1.1372e-01],
            [ 2.3924e-02,  1.2906e-03,  1.1404e-01],
            [ 2.3799e-02,  2.7901e-03,  1.1405e-01],
            [ 2.3700e-02,  2.8593e-03,  1.1403e-01],
            [ 2.2516e-02,  1.4079e-03,  1.1375e-01],
            [ 2.3358e-02,  3.4561e-03,  1.1397e-01],
            [ 2.4386e-02,  1.9534e-03,  1.1415e-01],
            [ 2.3621e-02,  2.1514e-03,  1.1400e-01],
            [ 2.3817e-02,  1.2542e-03,  1.1402e-01],
            [ 2.2500e-02,  3.3357e-03,  1.1379e-01],
            [ 2.3923e-02,  2.4151e-03,  1.1407e-01],
            [ 2.4372e-02,  2.7461e-03,  1.1417e-01]],

            [[ 2.0388e-02,  2.4630e-03,  1.2948e-01],
            [ 1.9897e-02,  2.1101e-03,  1.2934e-01],
            [ 1.8154e-02,  2.2393e-03,  1.2885e-01],
            [ 2.0566e-02,  2.6055e-03,  1.2954e-01],
            [ 1.9633e-02,  6.7339e-04,  1.2922e-01],
            [ 1.9092e-02,  1.7212e-03,  1.2910e-01],
            [ 1.8728e-02,  3.2463e-03,  1.2904e-01],
            [ 1.9587e-02,  2.5168e-03,  1.2926e-01],
            [ 1.8108e-02,  1.8125e-03,  1.2882e-01],
            [ 1.8303e-02,  2.7790e-03,  1.2890e-01],
            [ 1.9938e-02,  1.1587e-03,  1.2932e-01],
            [ 1.9055e-02,  3.3182e-03,  1.2913e-01],
            [ 1.8824e-02,  2.0922e-03,  1.2903e-01],
            [ 1.9151e-02,  3.2003e-03,  1.2916e-01],
            [ 1.9788e-02,  2.4894e-03,  1.2932e-01],
            [ 1.8767e-02,  2.4214e-03,  1.2902e-01],
            [ 1.9677e-02,  2.8752e-03,  1.2929e-01],
            [ 1.9597e-02,  3.0849e-03,  1.2928e-01],
            [ 1.9400e-02,  2.8054e-03,  1.2921e-01],
            [ 1.9838e-02,  2.3652e-03,  1.2933e-01]]])

    # print(cylinder_mesh_points)

    # # Plot cylinder_mesh_points using matplotlib
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(cylinder_mesh_points[:, :, 0], cylinder_mesh_points[:, :, 1], cylinder_mesh_points[:, :, 2])
    # ax.scatter(og_plot_points[:, 0], og_plot_points[:, 1], og_plot_points[:, 2], color='red')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Cylinder Mesh Points')

    # ax.set_xlim(-0.1, 0.1)
    # ax.set_ylim(-0.1, 0.1)
    # ax.set_zlim(-0.1, 0.1)

    # plt.tight_layout()
    # plt.show()

    # # Find L2 norm of all points in cylinder_mesh_points (in torch)
    # cylinder_mesh_points_norm = torch.norm(cylinder_mesh_points, dim=2)
    # print("cylinder_mesh_points_norm: ", cylinder_mesh_points_norm)

    # # Find average of each row in cylinder_mesh_points_norm (in torch)
    # cylinder_mesh_points_norm_avg = torch.mean(cylinder_mesh_points_norm, dim=1)
    # print("cylinder_mesh_points_norm_avg: ", cylinder_mesh_points_norm_avg)
