import numpy as np


def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y, z] a point on the first line
    a2: [x, y, z] another point on the first line
    b1: [x, y, z] a point on the second line
    b2: [x, y, z] another point on the second line
    """
    lines = np.vstack([a1, a2, b1, b2])  # s for stacked
    # h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous if the input is 2D
    l1 = np.cross(lines[0], lines[1])  # get first line
    l2 = np.cross(lines[2], lines[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    # if z == 0:                          # lines are parallel
    #     return (float('inf'), float('inf'))
    intersect_p = np.array([x, y, z])
    return intersect_p


gt_centline = np.array([[0.5, 0.5, 0.5], [1.8333901099811494, 0.49983521955662175, 0.5001647804433778],
                        [35.58591990879948, 6.191963209589745, -5.191963209589755],
                        [36.085861528232165, 7.06569109504143, -6.065691095041432]])

tangent_p0 = gt_centline[1, :] - gt_centline[0, :]
tangent_p1 = gt_centline[-1, :] - gt_centline[-2, :]

print()