import numpy as np
import scipy.optimize

import pdb

# def get_intersect(a1, a2, b1, b2):
#     """
#     Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
#     a1: [x, y, z] a point on the first line
#     a2: [x, y, z] another point on the first line
#     b1: [x, y, z] a point on the second line
#     b2: [x, y, z] another point on the second line
#     """
#     lines = np.vstack([a1, a2, b1, b2])  # s for stacked


# takes in two lines, the line formed by pt1 and pt2, and the line formed by pt3 and pt4, and finds their intersection or closest point
# please ref : https://stackoverflow.com/questions/44631259/line-line-intersection-in-python-with-numpy
# another ref (with analytical form) : https://stackoverflow.com/questions/2316490/the-algorithm-to-find-the-point-of-intersection-of-two-3d-line-segment
def get_intersect(pt1, pt2, pt3, pt4):
    #least squares method
    def errFunc(estimates):
        s, t = estimates
        x = pt1 + s * (pt2 - pt1) - (pt3 + t * (pt4 - pt3))
        return x

    estimates = [1, 1]

    sols = scipy.optimize.least_squares(errFunc, estimates)
    s, t = sols.x

    x1 = pt1[0] + s * (pt2[0] - pt1[0])
    x2 = pt3[0] + t * (pt4[0] - pt3[0])
    y1 = pt1[1] + s * (pt2[1] - pt1[1])
    y2 = pt3[1] + t * (pt4[1] - pt3[1])
    z1 = pt1[2] + s * (pt2[2] - pt1[2])
    z2 = pt3[2] + t * (pt4[2] - pt3[2])

    x = (x1 + x2) / 2  # halfway point if they don't match
    y = (y1 + y2) / 2  # halfway point if they don't match
    z = (z1 + z2) / 2  # halfway point if they don't match

    return (x, y, z)


gt_centline = np.array([[0.5, 0.5, 0.5], [1.8333901099811494, 0.49983521955662175, 0.5001647804433778],
                        [34.00667230795771, 4.680739199967527, -3.6807391999675203],
                        [35.58591990879948, 6.191963209589745, -5.191963209589755],
                        [36.085861528232165, 7.06569109504143, -6.065691095041432]])

tangent_p0 = gt_centline[1, :] - gt_centline[0, :]
tangent_p1 = gt_centline[-3, :] - gt_centline[-1, :]

p0 = gt_centline[0, :]
p0_vec = p0 + tangent_p0 / np.linalg.norm(tangent_p0, ord=None, axis=0, keepdims=False)
p1 = gt_centline[-1, :]
p1_vec = p1 + tangent_p0 / np.linalg.norm(tangent_p1, ord=None, axis=0, keepdims=False)

(x, y, z) = get_intersect(p0, p0_vec, p1, p1_vec)

print(x, y, z)

pdb.set_trace()