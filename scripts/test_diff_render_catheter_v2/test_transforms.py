import numpy as np

def cc_transform_1dof(p_0, phi, u, l, r, s=1):
    """
    Calculate constant curvature 1DoF transformation
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        phi (float): phi (float): (in radians) phi parameter
        u (float): tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    k = u / r

    T = np.array([
        [np.cos(phi) ** 2 * (np.cos(k * s) - 1) + 1,      np.sin(phi) * np.cos(phi) * (np.cos(k * s) - 1),        np.cos(phi) * np.sin(k * s), np.cos(phi) * (1 - np.cos(k * s)) / k],
        [np.sin(phi) * np.cos(phi) * (np.cos(k * s) - 1), np.cos(phi) ** 2 * (1 - np.cos(k * s)) + np.cos(k * s), np.sin(phi) * np.sin(k * s), np.sin(phi) * (1 - np.cos(k * s)) / k],
        [-1 * np.cos(phi) * np.sin(k * s),                -1 * np.sin(phi) * np.sin(k * s),                       np.cos(k * s),               np.sin(k * s) / k                    ],
        [0,                                               0,                                                      0,                           1                                    ]])

    return (T @ p_0_4d)[:3]


def cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    """
    Calculate constant curvature 3DoF transformation
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    u = np.sqrt(ux ** 2 + uy ** 2)
    k = u / r

    T = np.array([
        [1 + (ux ** 2) / (u ** 2) * (np.cos(k * s) - 1), ux * uy / (u ** 2) * (np.cos(k * s) - 1),       -1 * ux / u * np.sin(k * s), r * l * ux * (1 - np.cos(k * s)) / (u ** 2)],
        [ux * uy / (u ** 2) * (np.cos(k * s) - 1),       1 + (uy ** 2) / (u ** 2) * (np.cos(k * s) - 1), -1 * uy / u * np.sin(k * s), r * l * uy * (1 - np.cos(k * s)) / (u ** 2)],
        [ux / u * np.sin(k * s),                         uy / u * np.sin(k * s),                         np.cos(k * s),               r * l * np.sin(k * s) / u                  ],
        [0,                                              0,                                              0,                           1                                          ]], dtype=object)

    return (T @ p_0_4d)[:3]


def d_ux_cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    """
    Calculate derivative of constant curvature 3DoF transformation with respect to ux
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    u = np.sqrt(ux ** 2 + uy ** 2)
    
    dT_dux = np.array([
        [(-2 * ux**3 * (-1 + np.cos((s * u) / r))) / (u**4) + (2 * ux * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux**3 * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * ux**2 * uy * (-1 + np.cos((s * u) / r))) / (u**4) + (uy * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux**2 * uy * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * ux**2 * np.cos((s * u) / r)) / (r * (u**2))) + (ux**2 * np.sin((s * u) / r)) / (u**3) - np.sin((s * u) / r) / u,
         (-2 * l * r * ux**2 * (1 - np.cos((s * u) / r))) / (u**4) + (l * r * (1 - np.cos((s * u) / r))) / (u**2) + (l * s * ux**2 * np.sin((s * u) / r)) / (u**3)],
        [(-2 * ux**2 * uy * (-1 + np.cos((s * u) / r))) / (u**4) + (uy * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux**2 * uy * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * ux * uy**2 * (-1 + np.cos((s * u) / r))) / (u**4) - (s * ux * uy**2 * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * ux * uy * np.cos((s * u) / r)) / (r * (u**2))) + (ux * uy * np.sin((s * u) / r)) / (u**3),
         (-2 * l * r * ux * uy * (1 - np.cos((s * u) / r))) / (u**4) + (l * s * ux * uy * np.sin((s * u) / r)) / (u**3)],
        [(s * ux**2 * np.cos((s * u) / r)) / (r * (u**2)) - (ux**2 * np.sin((s * u) / r)) / (u**3) + np.sin((s * u) / r) / u,
         (s * ux * uy * np.cos((s * u) / r)) / (r * (u**2)) - (ux * uy * np.sin((s * u) / r)) / (u**3),
         -((s * ux * np.sin((s * u) / r)) / (r * u)),
         (l * s * ux * np.cos((s * u) / r)) / (u**2) - (l * r * ux * np.sin((s * u) / r)) / (u**3)],
        [0, 0, 0, 0]])
        
    return (dT_dux @ p_0_4d)[:3]


def d_uy_cc_transform_3dof(p_0, ux, uy, l, r, s=1):
    """
    Calculate derivative of constant curvature 3DoF transformation with respect to uy
    
    Args:
        p_0 ((3,) numpy array): start point of catheter
        ux (float): 1st pair of tendon length (responsible for catheter bending)
        uy (float): 2nd pair of tendon length (responsible for catheter bending)
        l (float): length of catheter
        r (float): cross section radius of catheter
        s (float from 0 to 1 inclusive): s value representing position on the CC curve
    """
    p_0_4d = np.append(p_0, 1)
    u = np.sqrt(ux ** 2 + uy ** 2)
    
    dT_duy = np.array([
        [(-2 * ux**2 * uy * (-1 + np.cos((s * u) / r))) / (u**4) - (s * ux**2 * uy * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * ux * uy**2 * (-1 + np.cos((s * u) / r))) / (u**4) + (ux * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux * uy**2 * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * ux * uy * np.cos((s * u) / r)) / (r * (u**2))) + (ux * uy * np.sin((s * u) / r)) / (u**3),
         (-2 * l * r * ux * uy * (1 - np.cos((s * u) / r))) / (u**4) + (l * s * ux * uy * np.sin((s * u) / r)) / (u**3)],
        [(-2 * ux * uy**2 * (-1 + np.cos((s * u) / r))) / (u**4) + (ux * (-1 + np.cos((s * u) / r))) / (u**2) - (s * ux * uy**2 * np.sin((s * u) / r)) / (r * (u**3)),
         (-2 * uy**3 * (-1 + np.cos((s * u) / r))) / (u**4) + (2 * uy * (-1 + np.cos((s * u) / r))) / (u**2) - (s * uy**3 * np.sin((s * u) / r)) / (r * (u**3)),
         -((s * uy**2 * np.cos((s * u) / r)) / (r * (u**2))) + (uy**2 * np.sin((s * u) / r)) / (u**3) - np.sin((s * u) / r) / u,
         (-2 * l * r * uy**2 * (1 - np.cos((s * u) / r))) / (u**4) + (l * r * (1 - np.cos((s * u) / r))) / (u**2) + (l * s * uy**2 * np.sin((s * u) / r)) / (u**3)],
        [(s * ux * uy * np.cos((s * u) / r)) / (r * (u**2)) - (ux * uy * np.sin((s * u) / r)) / (u**3),
         (s * uy**2 * np.cos((s * u) / r)) / (r * (u**2)) - (uy**2 * np.sin((s * u) / r)) / (u**3) + np.sin((s * u) / r) / u,
         -((s * uy * np.sin((s * u) / r)) / (r * u)),
         (l * s * uy * np.cos((s * u) / r)) / (u**2) - (l * r * uy * np.sin((s * u) / r)) / (u**3)],
        [0, 0, 0, 0]])
    
    return (dT_duy @ p_0_4d)[:3]