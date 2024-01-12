import numpy as np
import test_transforms

def d_bezier_d_p_start(s_bezier):
    """
    Calculate derivative of Bezier curve with respect to the start point

    Args:
        s_bezier (float from 0 to 1 inclusive): s value

    Returns:
        (float) derivative of Bezier curve with respect to the start point
    """
    return (s_bezier - 1) ** 2


def d_bezier_d_p_mid(s_bezier):
    """
    Calculate derivative of Bezier curve with respect to the middle control point

    Args:
        s_bezier (float from 0 to 1 inclusive): s value

    Returns:
        (float) derivative of Bezier curve with respect to the middle control point
    """
    return 4 * s_bezier * (1 - s_bezier)


def d_bezier_d_p_end(s_bezier):
    """
    Calculate derivative of Bezier curve with respect to the end point

    Args:
        s_bezier (float from 0 to 1 inclusive): s value

    Returns:
        (float) derivative of Bezier curve with respect to the end point
    """
    return s_bezier * (3 * s_bezier - 2)


def calculate_jacobian_2dof_ux_uy(p_start, ux, uy, l, r):
    """
    Calculate Jacobian of 2DoF interspace control with (ux, uy) parameterization

    Args:
        p_start ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter

    Returns:
        ((6, 2) numpy array): Jacobian of 2DoF interspace control with (ux, uy) parameterization
    """
    bezier_mid_over_cc_start = d_bezier_d_p_start(0.5)
    bezier_end_over_cc_start = d_bezier_d_p_start(1.0)
    bezier_mid_over_cc_mid = d_bezier_d_p_mid(0.5)
    bezier_end_over_cc_mid = d_bezier_d_p_mid(1.0)
    bezier_mid_over_cc_end = d_bezier_d_p_end(0.5)
    bezier_end_over_cc_end = d_bezier_d_p_end(1.0)

    cc_start_over_ux = test_transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_start_over_uy = test_transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.0)
    cc_mid_over_ux = test_transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_mid_over_uy = test_transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=0.5)
    cc_end_over_ux = test_transforms.d_ux_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)
    cc_end_over_uy = test_transforms.d_uy_cc_transform_3dof(p_start, ux, uy, l, r, s=1.0)


    G_p_mid_ux = bezier_mid_over_cc_start * cc_start_over_ux + bezier_mid_over_cc_mid * cc_mid_over_ux + bezier_mid_over_cc_end * cc_end_over_ux
    G_p_mid_uy = bezier_mid_over_cc_start * cc_start_over_uy + bezier_mid_over_cc_mid * cc_mid_over_uy + bezier_mid_over_cc_end * cc_end_over_uy
    G_p_end_ux = bezier_end_over_cc_start * cc_start_over_ux + bezier_end_over_cc_mid * cc_mid_over_ux + bezier_end_over_cc_end * cc_end_over_ux
    G_p_end_uy = bezier_end_over_cc_start * cc_start_over_uy + bezier_end_over_cc_mid * cc_mid_over_uy + bezier_end_over_cc_end * cc_end_over_uy

    J = np.zeros((6, 2))
    J[0:3, 0] = G_p_mid_ux
    J[0:3, 1] = G_p_mid_uy
    J[3:6, 0] = G_p_end_ux
    J[3:6, 1] = G_p_end_uy

    return J

