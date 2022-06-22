import numpy as np
import pdb

import cv2
from skimage.morphology import skeletonize

from datetime import datetime
from datetime import timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


def getContourSamples(self):

    res_width = 640
    res_height = 480

    img_path = "/home/fei/ARCLab-CCCatheter/data/rendered_images/dof2_64/dof2_c16_-0.0005_-0.005_0.2_0.01.png"

    raw_img_rgb = cv2.imread(img_path)
    downscale = 1.0
    raw_img_rgb = cv2.resize(raw_img_rgb,
                             (int(raw_img_rgb.shape[1] / downscale), int(raw_img_rgb.shape[0] / downscale)))
    raw_img = cv2.cvtColor(raw_img_rgb, cv2.COLOR_RGB2GRAY)

    ret, img_thresh = cv2.threshold(raw_img.copy(), 80, 255, cv2.THRESH_BINARY)

    # img_thresh = cv2.bitwise_not(img_thresh)
    img_thresh = img_thresh

    # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    # ax = axes.ravel()
    # ax[0].imshow(raw_img, cmap=cm.gray)
    # ax[0].set_title('Input image')
    # ax[1].imshow(img_thresh, cmap=cm.gray)
    # ax[1].set_title('img_thresh image')
    # plt.show()

    # img_thresh = cv2.adaptiveThreshold(raw_img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # print(raw_img)

    # perform skeletonization, need to extend the boundary of the image
    extend_dim = int(60)

    img_thresh_extend = np.zeros((res_height, res_width + extend_dim))
    img_thresh_extend[0:res_height, 0:res_width] = img_thresh.copy() / 255

    left_boundarylineA_id = np.squeeze(np.argwhere(img_thresh_extend[:, res_width - 1]))
    left_boundarylineB_id = np.squeeze(np.argwhere(img_thresh_extend[:, res_width - 10]))

    extend_vec_pt1_center = np.array([res_width, (left_boundarylineA_id[0] + left_boundarylineA_id[-1]) / 2])
    extend_vec_pt2_center = np.array([res_width - 5, (left_boundarylineB_id[0] + left_boundarylineB_id[-1]) / 2])
    exten_vec = extend_vec_pt2_center - extend_vec_pt1_center

    if exten_vec[1] == 0:
        exten_vec[1] += 0.00000001

    k_extend = exten_vec[0] / exten_vec[1]
    b_extend_up = res_width - k_extend * left_boundarylineA_id[0]
    b_extend_dw = res_width - k_extend * left_boundarylineA_id[-1]

    # then it could be able to get the intersection point with boundary
    extend_ROI = np.array([
        np.array([res_width, left_boundarylineA_id[0]]),
        np.array([res_width, left_boundarylineA_id[-1]]),
        np.array([res_width + extend_dim,
                  int(((res_width + extend_dim) - b_extend_dw) / k_extend)]),
        np.array([res_width + extend_dim,
                  int(((res_width + extend_dim) - b_extend_up) / k_extend)])
    ])

    img_thresh_extend = cv2.fillPoly(img_thresh_extend, [extend_ROI], 1)

    skeleton = skeletonize(img_thresh_extend)

    img_raw_skeleton = np.argwhere(skeleton[:, 0:res_width] == 1)


np.random.seed(1991)

start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005), ConstantVelocity(0.005)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

truth_data = np.array(
    [639.0000, 241.0072], [618.9486, 239.9070], [548.5718, 235.4676], [494.5704, 231.5884], [450.8459, 228.0558],
    [413.9810, 224.7499], [381.9123, 221.5984], [353.3202, 218.5549], [327.3211, 215.5882], [303.3021, 212.6769],
    [280.8234, 209.8053], [259.5619, 206.9621], [239.2745, 204.1387], [219.7751, 201.3284], [200.9182, 198.5261],
    [182.5888, 195.7277], [168.2423, 193.4894], [164.1701, 194.3716], [161.5956, 197.7816], [161.6741, 202.1046],
    [164.3690, 205.3178], [166.1060, 205.9440], [180.3879, 208.6722], [198.6306, 212.1446], [217.3936, 215.7003],
    [236.7907, 219.3583], [256.9660, 223.1424], [278.1040, 227.0834], [300.4454, 231.2213], [324.3101, 235.6089],
    [350.1330, 240.3187], [378.5208, 245.4512], [410.3461, 251.1512], [446.9128, 257.6350], [490.2577, 265.2406],
    [543.7531, 274.5273], [613.4150, 286.4933], [639.0000, 290.8440], [639.0000, 280.8767], [639.0000, 270.9093],
    [639.0000, 260.9420], [639.0000, 250.9746], [639.0000, 241.0072])

for k in range(1, 21):
    truth.append(
        GroundTruthState(transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                         timestamp=start_time + timedelta(seconds=k)))

pdb.set_trace()