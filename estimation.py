import cv2
import numpy as np
# from scipy.optimize import minimize


def org_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Original estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    mse_min = np.inf
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    DC_border = np.array([-1024, 1024]) // Q[0, 0]
    DC_range = np.arange(DC_border[0], DC_border[1]+1)
    for dc in DC_range:
        mse1 = 0.
        mse2 = 0.
        mse3 = 0.
        dct_target[0, 0] = dc
        spatial_target = cv2.idct(dct_target * Q) + 128
        if dct_left is not False:
            mse1 += np.sum(np.square((spatial_left[0:8, 7] - spatial_target[0:8, 0]) -
                                     (spatial_left[0:8, 6] - spatial_left[0:8, 7]))) / 8
            mse2 += np.sum(np.square((spatial_left[0:7, 7] - spatial_target[1:8, 0]) -
                                     (spatial_left[0:7, 6] - spatial_left[1:8, 7]))) / 7
            mse3 += np.sum(np.square((spatial_left[1:8, 7] - spatial_target[0:7, 0]) -
                                     (spatial_left[1:8, 6] - spatial_left[0:7, 7]))) / 7
        if dct_up is not False:
            mse1 += np.sum(np.square((spatial_up[7, 0:8] - spatial_target[0, 0:8]) -
                                     (spatial_up[6, 0:8] - spatial_up[7, 0:8]))) / 8
            mse2 += np.sum(np.square((spatial_up[7, 0:7] - spatial_target[0, 1:8]) -
                                     (spatial_up[6, 0:7] - spatial_up[7, 1:8]))) / 7
            mse3 += np.sum(np.square((spatial_up[7, 1:8] - spatial_target[0, 0:7]) -
                                     (spatial_up[6, 1:8] - spatial_up[7, 0:7]))) / 7
        if dct_right is not False:
            mse1 += np.sum(np.square((spatial_right[0:8, 0] - spatial_target[0:8, 7]) -
                                     (spatial_right[0:8, 1] - spatial_right[0:8, 0]))) / 8
            mse2 += np.sum(np.square((spatial_right[0:7, 0] - spatial_target[1:8, 7]) -
                                     (spatial_right[0:7, 1] - spatial_right[1:8, 0]))) / 7
            mse3 += np.sum(np.square((spatial_right[1:8, 0] - spatial_target[0:7, 7]) -
                                     (spatial_right[1:8, 1] - spatial_right[0:7, 0]))) / 7
        if dct_down is not False:
            mse1 += np.sum(np.square((spatial_down[0, 0:8] - spatial_target[7, 0:8]) -
                                     (spatial_down[1, 0:8] - spatial_down[0, 0:8]))) / 8
            mse2 += np.sum(np.square((spatial_down[0, 0:7] - spatial_target[7, 1:8]) -
                                     (spatial_down[1, 0:7] - spatial_down[0, 1:8]))) / 7
            mse3 += np.sum(np.square((spatial_down[0, 1:8] - spatial_target[7, 0:7]) -
                                     (spatial_down[1, 1:8] - spatial_down[0, 0:7]))) / 7
        mse = np.min([mse1, mse2, mse3])
        if mse < mse_min:
            mse_min = mse
            dc_optimal = dc

    return dc_optimal


def mod_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Modified estimation by Qiu et al. of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    
    dc = []
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1, mse2, mse3 = 0., 0., 0.
        mse1 += np.sum((spatial_left[0:8, 7] - Q[0][0] * dct_left[0,0]/8.) - spatial_target[0:8, 0]) / 8
        mse2 += np.sum((spatial_left[0:7, 7] - Q[0][0] * dct_left[0,0]/8.) - spatial_target[1:8, 0]) / 7
        mse3 += np.sum((spatial_left[1:8, 7] - Q[0][0] * dct_left[0,0]/8.) - spatial_target[0:7, 0]) / 7
        # dc.append(dct_left[0,0] + 8 * (mse1 + mse2 + mse3) / (3 * Q[0][0]))
        mse = mse1 if abs(mse1) < abs(mse2) else mse2
        mse = mse if abs(mse) < abs(mse3) else mse3
        dc.append(dct_left[0,0] + 8 * mse / Q[0][0])

    if dct_up is not False:
        mse1, mse2, mse3 = 0., 0., 0.
        mse1 += np.sum((spatial_up[7, 0:8] - Q[0][0] * dct_up[0,0]/8.) - spatial_target[0, 0:8]) / 8
        mse2 += np.sum((spatial_up[7, 0:7] - Q[0][0] * dct_up[0,0]/8.) - spatial_target[0, 1:8]) / 7
        mse3 += np.sum((spatial_up[7, 1:8] - Q[0][0] * dct_up[0,0]/8.) - spatial_target[0, 0:7]) / 7
        # dc.append(dct_up[0,0] + 8 * (mse1 + mse2 + mse3) / (3 * Q[0][0]))
        mse = mse1 if abs(mse1) < abs(mse2) else mse2
        mse = mse if abs(mse) < abs(mse3) else mse3
        dc.append(dct_up[0,0] + 8 * mse / Q[0][0])

    if dct_right is not False:
        mse1, mse2, mse3 = 0., 0., 0.
        mse1 += np.sum((spatial_right[0:8, 0] - Q[0][0] * dct_right[0,0]/8.) - spatial_target[0:8, 7]) / 8
        mse2 += np.sum((spatial_right[0:7, 0] - Q[0][0] * dct_right[0,0]/8.) - spatial_target[1:8, 7]) / 7
        mse3 += np.sum((spatial_right[1:8, 0] - Q[0][0] * dct_right[0,0]/8.) - spatial_target[0:7, 7]) / 7
        # dc.append(dct_right[0,0] + 8 * (mse1 + mse2 + mse3) / (3 * Q[0][0]))
        mse = mse1 if abs(mse1) < abs(mse2) else mse2
        mse = mse if abs(mse) < abs(mse3) else mse3
        dc.append(dct_right[0,0] + 8 * mse / Q[0][0])

    if dct_down is not False:
        mse1, mse2, mse3 = 0., 0., 0.
        mse1 += np.sum((spatial_down[0, 0:8] - Q[0][0] * dct_down[0,0]/8.) - spatial_target[7, 0:8]) / 8
        mse2 += np.sum((spatial_down[0, 0:7] - Q[0][0] * dct_down[0,0]/8.) - spatial_target[7, 1:8]) / 7
        mse3 += np.sum((spatial_down[0, 1:8] - Q[0][0] * dct_down[0,0]/8.) - spatial_target[7, 0:7]) / 7
        # dc.append(dct_down[0,0] + 8 * (mse1 + mse2 + mse3) / (3 * Q[0][0]))
        mse = mse1 if abs(mse1) < abs(mse2) else mse2
        mse = mse if abs(mse) < abs(mse3) else mse3
        dc.append(dct_down[0,0] + 8 * mse / Q[0][0])

    target_dc = 0.
    for i in dc:
        target_dc += i
    target_dc /= len(dc)

    return target_dc


def mod_spd_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    count1, count2, count3 = 0, 0, 0
    diff1, diff2, diff3 = 0., 0., 0.
    if dct_left is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_left[0:8, 7] + spatial_left[0:8, 7] - spatial_left[0:8, 6] - spatial_target[0:8, 0]
        diff2 += spatial_left[0:7, 7] + spatial_left[1:8, 7] - spatial_left[0:7, 6] - spatial_target[1:8, 0]
        diff3 += spatial_left[1:8, 7] + spatial_left[0:7, 7] - spatial_left[1:8, 6] - spatial_target[0:7, 0]

    if dct_up is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_up[7, 0:8] + spatial_up[7, 0:8] - spatial_up[6, 0:8] - spatial_target[0, 0:8]
        diff2 += spatial_up[7, 0:7] + spatial_up[7, 1:8] - spatial_up[6, 0:7] - spatial_target[0, 1:8]
        diff3 += spatial_up[7, 1:8] + spatial_up[7, 0:7] - spatial_up[6, 1:8] - spatial_target[0, 0:7]

    if dct_right is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_right[0:8, 0] + spatial_right[0:8, 0] - spatial_right[0:8, 1] - spatial_target[0:8, 7]
        diff2 += spatial_right[0:7, 0] + spatial_right[1:8, 0] - spatial_right[0:7, 1] - spatial_target[1:8, 7]
        diff3 += spatial_right[1:8, 0] + spatial_right[0:7, 0] - spatial_right[1:8, 1] - spatial_target[0:7, 7]

    if dct_down is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_down[0, 0:8] + spatial_down[0, 0:8] - spatial_down[1, 0:8] - spatial_target[7, 0:8]
        diff2 += spatial_down[0, 0:7] + spatial_down[0, 1:8] - spatial_down[1, 0:7] - spatial_target[7, 1:8]
        diff3 += spatial_down[0, 1:8] + spatial_down[0, 0:7] - spatial_down[1, 1:8] - spatial_target[7, 0:7]

    dc1 = np.clip(np.round(8 * np.sum(diff1) / (count1 * Q[0][0])), DC_border[0], DC_border[1])
    dc2 = np.clip(np.round(8 * np.sum(diff2) / (count2 * Q[0][0])), DC_border[0], DC_border[1])
    dc3 = np.clip(np.round(8 * np.sum(diff3) / (count3 * Q[0][0])), DC_border[0], DC_border[1])

    mse1 = 0.
    mse2 = 0.
    mse3 = 0.
    dct_target[0, 0] = dc1
    spatial_target1 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc2
    spatial_target2 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc3
    spatial_target3 = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1 += np.sum(np.square((spatial_left[0:8, 7] - spatial_target1[0:8, 0]) -
                                    (spatial_left[0:8, 6] - spatial_left[0:8, 7]))) / 8
        mse2 += np.sum(np.square((spatial_left[0:7, 7] - spatial_target2[1:8, 0]) -
                                    (spatial_left[0:7, 6] - spatial_left[1:8, 7]))) / 7
        mse3 += np.sum(np.square((spatial_left[1:8, 7] - spatial_target3[0:7, 0]) -
                                    (spatial_left[1:8, 6] - spatial_left[0:7, 7]))) / 7
    if dct_up is not False:
        mse1 += np.sum(np.square((spatial_up[7, 0:8] - spatial_target1[0, 0:8]) -
                                    (spatial_up[6, 0:8] - spatial_up[7, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_up[7, 0:7] - spatial_target2[0, 1:8]) -
                                    (spatial_up[6, 0:7] - spatial_up[7, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_up[7, 1:8] - spatial_target3[0, 0:7]) -
                                    (spatial_up[6, 1:8] - spatial_up[7, 0:7]))) / 7
    if dct_right is not False:
        mse1 += np.sum(np.square((spatial_right[0:8, 0] - spatial_target1[0:8, 7]) -
                                    (spatial_right[0:8, 1] - spatial_right[0:8, 0]))) / 8
        mse2 += np.sum(np.square((spatial_right[0:7, 0] - spatial_target2[1:8, 7]) -
                                    (spatial_right[0:7, 1] - spatial_right[1:8, 0]))) / 7
        mse3 += np.sum(np.square((spatial_right[1:8, 0] - spatial_target3[0:7, 7]) -
                                    (spatial_right[1:8, 1] - spatial_right[0:7, 0]))) / 7
    if dct_down is not False:
        mse1 += np.sum(np.square((spatial_down[0, 0:8] - spatial_target1[7, 0:8]) -
                                    (spatial_down[1, 0:8] - spatial_down[0, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_down[0, 0:7] - spatial_target2[7, 1:8]) -
                                    (spatial_down[1, 0:7] - spatial_down[0, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_down[0, 1:8] - spatial_target3[7, 0:7]) -
                                    (spatial_down[1, 1:8] - spatial_down[0, 0:7]))) / 7
    
    dc_optimal = dc1 if mse1 < min(mse2, mse3) else dc2 if mse2 < min(mse1, mse3) else dc3
    return dc_optimal


def ehc_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    count1, count2, count3 = 0, 0, 0
    diff1, diff2, diff3 = 0., 0., 0.
    if dct_left is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_left[0:8, 6] - spatial_left[0:8, 7])
        diff2 += (spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_left[0:7, 6] - spatial_left[1:8, 7])
        diff3 += (spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_left[1:8, 6] - spatial_left[0:7, 7])

    if dct_up is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_up[6, 0:8] - spatial_up[7, 0:8])
        diff2 += (spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_up[6, 0:7] - spatial_up[7, 1:8])
        diff3 += (spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_up[6, 1:8] - spatial_up[7, 0:7])

    if dct_right is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_right[0:8, 1] - spatial_right[0:8, 0])
        diff2 += (spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_right[0:7, 1] - spatial_right[1:8, 0])
        diff3 += (spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_right[1:8, 1] - spatial_right[0:7, 0])

    if dct_down is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_down[1, 0:8] - spatial_down[0, 0:8])
        diff2 += (spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_down[1, 0:7] - spatial_down[0, 1:8])
        diff3 += (spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_down[1, 1:8] - spatial_down[0, 0:7])

    dc1 = np.clip(8 * np.sum(diff1) / (count1 * Q[0][0]), DC_border[0], DC_border[1])
    dc2 = np.clip(8 * np.sum(diff2) / (count2 * Q[0][0]), DC_border[0], DC_border[1])
    dc3 = np.clip(8 * np.sum(diff3) / (count3 * Q[0][0]), DC_border[0], DC_border[1])

    mse1 = 0.
    mse2 = 0.
    mse3 = 0.
    dct_target[0, 0] = dc1
    spatial_target1 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc2
    spatial_target2 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc3
    spatial_target3 = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1 += np.sum(np.square((spatial_left[0:8, 7] - spatial_target1[0:8, 0]) - (spatial_left[0:8, 6] - spatial_left[0:8, 7]))) / 8
        mse2 += np.sum(np.square((spatial_left[0:7, 7] - spatial_target2[1:8, 0]) - (spatial_left[0:7, 6] - spatial_left[1:8, 7]))) / 7
        mse3 += np.sum(np.square((spatial_left[1:8, 7] - spatial_target3[0:7, 0]) - (spatial_left[1:8, 6] - spatial_left[0:7, 7]))) / 7
    if dct_up is not False:
        mse1 += np.sum(np.square((spatial_up[7, 0:8] - spatial_target1[0, 0:8]) - (spatial_up[6, 0:8] - spatial_up[7, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_up[7, 0:7] - spatial_target2[0, 1:8]) - (spatial_up[6, 0:7] - spatial_up[7, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_up[7, 1:8] - spatial_target3[0, 0:7]) - (spatial_up[6, 1:8] - spatial_up[7, 0:7]))) / 7
    if dct_right is not False:
        mse1 += np.sum(np.square((spatial_right[0:8, 0] - spatial_target1[0:8, 7]) - (spatial_right[0:8, 1] - spatial_right[0:8, 0]))) / 8
        mse2 += np.sum(np.square((spatial_right[0:7, 0] - spatial_target2[1:8, 7]) - (spatial_right[0:7, 1] - spatial_right[1:8, 0]))) / 7
        mse3 += np.sum(np.square((spatial_right[1:8, 0] - spatial_target3[0:7, 7]) - (spatial_right[1:8, 1] - spatial_right[0:7, 0]))) / 7
    if dct_down is not False:
        mse1 += np.sum(np.square((spatial_down[0, 0:8] - spatial_target1[7, 0:8]) - (spatial_down[1, 0:8] - spatial_down[0, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_down[0, 0:7] - spatial_target2[7, 1:8]) - (spatial_down[1, 0:7] - spatial_down[0, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_down[0, 1:8] - spatial_target3[7, 0:7]) - (spatial_down[1, 1:8] - spatial_down[0, 0:7]))) / 7
    
    dc_optimal = dc1 if mse1 < min(mse2, mse3) else dc2 if mse2 < min(mse1, mse3) else dc3
    return dc_optimal


def cen_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    count1, count2, count3 = 0, 0, 0
    diff1, diff2, diff3 = 0., 0., 0.
    if dct_left is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += ((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_left[0:8, 6] - spatial_left[0:8, 7])) - ((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_target[0:8, 0] - spatial_target[0:8, 1]))
        diff2 += ((spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_left[0:7, 6] - spatial_left[1:8, 7])) - ((spatial_left[0:7, 7] - spatial_target[0:7, 0]) - (spatial_target[1:8, 0] - spatial_target[1:8, 1]))
        diff3 += ((spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_left[1:8, 6] - spatial_left[0:7, 7])) - ((spatial_left[1:8, 7] - spatial_target[1:8, 0]) - (spatial_target[0:7, 0] - spatial_target[0:7, 1]))

    if dct_up is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += ((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_up[6, 0:8] - spatial_up[7, 0:8])) - ((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_target[0, 0:8] - spatial_target[1, 0:8]))
        diff2 += ((spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_up[6, 0:7] - spatial_up[7, 1:8])) - ((spatial_up[7, 0:7] - spatial_target[0, 0:7]) - (spatial_target[0, 1:8] - spatial_target[1, 1:8]))
        diff3 += ((spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_up[6, 1:8] - spatial_up[7, 0:7])) - ((spatial_up[7, 1:8] - spatial_target[0, 1:8]) - (spatial_target[0, 0:7] - spatial_target[1, 0:7]))

    if dct_right is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += ((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_right[0:8, 1] - spatial_right[0:8, 0])) - ((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_target[0:8, 7] - spatial_target[0:8, 6]))
        diff2 += ((spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_right[0:7, 1] - spatial_right[1:8, 0])) - ((spatial_right[0:7, 0] - spatial_target[0:7, 7]) - (spatial_target[1:8, 7] - spatial_target[1:8, 6]))
        diff3 += ((spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_right[1:8, 1] - spatial_right[0:7, 0])) - ((spatial_right[1:8, 0] - spatial_target[1:8, 7]) - (spatial_target[0:7, 7] - spatial_target[0:7, 6]))

    if dct_down is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += ((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_down[1, 0:8] - spatial_down[0, 0:8])) - ((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_target[7, 0:8] - spatial_target[6, 0:8]))
        diff2 += ((spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_down[1, 0:7] - spatial_down[0, 1:8])) - ((spatial_down[0, 0:7] - spatial_target[7, 0:7]) - (spatial_target[7, 1:8] - spatial_target[6, 1:8]))
        diff3 += ((spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_down[1, 1:8] - spatial_down[0, 0:7])) - ((spatial_down[0, 1:8] - spatial_target[7, 1:8]) - (spatial_target[7, 0:7] - spatial_target[6, 0:7]))

    dc1 = np.clip(8 * np.sum(diff1) / (count1 * Q[0][0]), DC_border[0], DC_border[1])
    dc2 = np.clip(8 * np.sum(diff2) / (count2 * Q[0][0]), DC_border[0], DC_border[1])
    dc3 = np.clip(8 * np.sum(diff3) / (count3 * Q[0][0]), DC_border[0], DC_border[1])

    mse1 = 0.
    mse2 = 0.
    mse3 = 0.
    dct_target[0, 0] = dc1
    spatial_target1 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc2
    spatial_target2 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc3
    spatial_target3 = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1 += np.sum(np.square(((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_left[0:8, 6] - spatial_left[0:8, 7])) - ((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_target[0:8, 0] - spatial_target[0:8, 1])))) / 8
        mse2 += np.sum(np.square(((spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_left[0:7, 6] - spatial_left[1:8, 7])) - ((spatial_left[0:7, 7] - spatial_target[0:7, 0]) - (spatial_target[1:8, 0] - spatial_target[1:8, 1])))) / 7
        mse3 += np.sum(np.square(((spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_left[1:8, 6] - spatial_left[0:7, 7])) - ((spatial_left[1:8, 7] - spatial_target[1:8, 0]) - (spatial_target[0:7, 0] - spatial_target[0:7, 1])))) / 7
    if dct_up is not False:
        mse1 += np.sum(np.square(((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_up[6, 0:8] - spatial_up[7, 0:8])) - ((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_target[0, 0:8] - spatial_target[1, 0:8])))) / 8
        mse2 += np.sum(np.square(((spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_up[6, 0:7] - spatial_up[7, 1:8])) - ((spatial_up[7, 0:7] - spatial_target[0, 0:7]) - (spatial_target[0, 1:8] - spatial_target[1, 1:8])))) / 7
        mse3 += np.sum(np.square(((spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_up[6, 1:8] - spatial_up[7, 0:7])) - ((spatial_up[7, 1:8] - spatial_target[0, 1:8]) - (spatial_target[0, 0:7] - spatial_target[1, 0:7])))) / 7
    if dct_right is not False:
        mse1 += np.sum(np.square(((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_right[0:8, 1] - spatial_right[0:8, 0])) - ((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_target[0:8, 7] - spatial_target[0:8, 6])))) / 8
        mse2 += np.sum(np.square(((spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_right[0:7, 1] - spatial_right[1:8, 0])) - ((spatial_right[0:7, 0] - spatial_target[0:7, 7]) - (spatial_target[1:8, 7] - spatial_target[1:8, 6])))) / 7
        mse3 += np.sum(np.square(((spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_right[1:8, 1] - spatial_right[0:7, 0])) - ((spatial_right[1:8, 0] - spatial_target[1:8, 7]) - (spatial_target[0:7, 7] - spatial_target[0:7, 6])))) / 7
    if dct_down is not False:
        mse1 += np.sum(np.square(((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_down[1, 0:8] - spatial_down[0, 0:8])) - ((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_target[7, 0:8] - spatial_target[6, 0:8])))) / 8
        mse2 += np.sum(np.square(((spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_down[1, 0:7] - spatial_down[0, 1:8])) - ((spatial_down[0, 0:7] - spatial_target[7, 0:7]) - (spatial_target[7, 1:8] - spatial_target[6, 1:8])))) / 7
        mse3 += np.sum(np.square(((spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_down[1, 1:8] - spatial_down[0, 0:7])) - ((spatial_down[0, 1:8] - spatial_target[7, 1:8]) - (spatial_target[7, 0:7] - spatial_target[6, 0:7])))) / 7
    
    dc_optimal = dc1 if mse1 < min(mse2, mse3) else dc2 if mse2 < min(mse1, mse3) else dc3
    return dc_optimal


def mmod_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    count1, count2, count3 = 0, 0, 0
    diff1, diff2, diff3 = 0., 0., 0.
    if dct_left is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_left[0:8, 7] - spatial_left[0:8, 6] + spatial_left[0:8, 7] - spatial_target[0:8, 0] + spatial_target[0:8, 1] - spatial_target[0:8, 0]
        diff2 += spatial_left[1:8, 7] - spatial_left[0:7, 6] + spatial_left[0:7, 7] - spatial_target[1:8, 0] + spatial_target[1:8, 1] - spatial_target[0:7, 0]
        diff3 += spatial_left[0:7, 7] - spatial_left[1:8, 6] + spatial_left[1:8, 7] - spatial_target[0:7, 0] + spatial_target[0:7, 1] - spatial_target[1:8, 0]

    if dct_up is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_up[7, 0:8] - spatial_up[6, 0:8] + spatial_up[7, 0:8] - spatial_target[0, 0:8] + spatial_target[1, 0:8] - spatial_target[0, 0:8]
        diff2 += spatial_up[7, 1:8] - spatial_up[6, 0:7] + spatial_up[7, 0:7] - spatial_target[0, 1:8] + spatial_target[1, 1:8] - spatial_target[0, 0:7]
        diff3 += spatial_up[7, 0:7] - spatial_up[6, 1:8] + spatial_up[7, 1:8] - spatial_target[0, 0:7] + spatial_target[1, 0:7] - spatial_target[0, 1:8]

    if dct_right is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_right[0:8, 0] - spatial_right[0:8, 1] + spatial_right[0:8, 0] - spatial_target[0:8, 7] + spatial_target[0:8, 6] - spatial_target[0:8, 7]
        diff2 += spatial_right[1:8, 0] - spatial_right[0:7, 1] + spatial_right[0:7, 0] - spatial_target[1:8, 7] + spatial_target[1:8, 6] - spatial_target[0:7, 7]
        diff3 += spatial_right[0:7, 0] - spatial_right[1:8, 1] + spatial_right[1:8, 0] - spatial_target[0:7, 7] + spatial_target[0:7, 6] - spatial_target[1:8, 7]

    if dct_down is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_down[0, 0:8] - spatial_down[1, 0:8] + spatial_down[0, 0:8] - spatial_target[7, 0:8] + spatial_target[6, 0:8] - spatial_target[7, 0:8]
        diff2 += spatial_down[0, 1:8] - spatial_down[1, 0:7] + spatial_down[0, 0:7] - spatial_target[7, 1:8] + spatial_target[6, 1:8] - spatial_target[7, 0:7]
        diff3 += spatial_down[0, 0:7] - spatial_down[1, 1:8] + spatial_down[0, 1:8] - spatial_target[7, 0:7] + spatial_target[6, 0:7] - spatial_target[7, 1:8]

    dc1 = np.clip(8 * np.sum(diff1) / (count1 * Q[0][0]), DC_border[0], DC_border[1])
    dc2 = np.clip(8 * np.sum(diff2) / (count2 * Q[0][0]), DC_border[0], DC_border[1])
    dc3 = np.clip(8 * np.sum(diff3) / (count3 * Q[0][0]), DC_border[0], DC_border[1])
    # dc1 = np.clip(np.round(8 * np.sum(diff1) / (count1 * Q[0][0])), DC_border[0], DC_border[1])
    # dc2 = np.clip(np.round(8 * np.sum(diff2) / (count2 * Q[0][0])), DC_border[0], DC_border[1])
    # dc3 = np.clip(np.round(8 * np.sum(diff3) / (count3 * Q[0][0])), DC_border[0], DC_border[1])

    mse1 = 0.
    mse2 = 0.
    mse3 = 0.
    dct_target[0, 0] = dc1
    spatial_target1 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc2
    spatial_target2 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc3
    spatial_target3 = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1 += np.sum(np.square(spatial_left[0:8, 7] - spatial_left[0:8, 6] + spatial_left[0:8, 7] - spatial_target1[0:8, 0] + spatial_target1[0:8, 1] - spatial_target1[0:8, 0])) / 8
        mse2 += np.sum(np.square(spatial_left[1:8, 7] - spatial_left[0:7, 6] + spatial_left[0:7, 7] - spatial_target2[1:8, 0] + spatial_target2[1:8, 1] - spatial_target2[0:7, 0])) / 7
        mse3 += np.sum(np.square(spatial_left[0:7, 7] - spatial_left[1:8, 6] + spatial_left[1:8, 7] - spatial_target3[0:7, 0] + spatial_target3[0:7, 1] - spatial_target3[1:8, 0])) / 7
    if dct_up is not False:
        mse1 += np.sum(np.square(spatial_up[7, 0:8] - spatial_up[6, 0:8] + spatial_up[7, 0:8] - spatial_target1[0, 0:8] + spatial_target1[1, 0:8] - spatial_target1[0, 0:8])) / 8
        mse2 += np.sum(np.square(spatial_up[7, 1:8] - spatial_up[6, 0:7] + spatial_up[7, 0:7] - spatial_target2[0, 1:8] + spatial_target2[1, 1:8] - spatial_target2[0, 0:7])) / 7
        mse3 += np.sum(np.square(spatial_up[7, 0:7] - spatial_up[6, 1:8] + spatial_up[7, 1:8] - spatial_target3[0, 0:7] + spatial_target3[1, 0:7] - spatial_target3[0, 1:8])) / 7
    if dct_right is not False:
        mse1 += np.sum(np.square(spatial_right[0:8, 0] - spatial_right[0:8, 1] + spatial_right[0:8, 0] - spatial_target1[0:8, 7] + spatial_target1[0:8, 6] - spatial_target1[0:8, 7])) / 8
        mse2 += np.sum(np.square(spatial_right[1:8, 0] - spatial_right[0:7, 1] + spatial_right[0:7, 0] - spatial_target2[1:8, 7] + spatial_target2[1:8, 6] - spatial_target2[0:7, 7])) / 7
        mse3 += np.sum(np.square(spatial_right[0:7, 0] - spatial_right[1:8, 1] + spatial_right[1:8, 0] - spatial_target3[0:7, 7] + spatial_target3[0:7, 6] - spatial_target3[1:8, 7])) / 7
    if dct_down is not False:
        mse1 += np.sum(np.square(spatial_down[0, 0:8] - spatial_down[1, 0:8] + spatial_down[0, 0:8] - spatial_target1[7, 0:8] + spatial_target1[6, 0:8] - spatial_target1[7, 0:8])) / 8
        mse2 += np.sum(np.square(spatial_down[0, 1:8] - spatial_down[1, 0:7] + spatial_down[0, 0:7] - spatial_target2[7, 1:8] + spatial_target2[6, 1:8] - spatial_target2[7, 0:7])) / 7
        mse3 += np.sum(np.square(spatial_down[0, 0:7] - spatial_down[1, 1:8] + spatial_down[0, 1:8] - spatial_target3[7, 0:7] + spatial_target3[6, 0:7] - spatial_target3[7, 1:8])) / 7
    
    dc_optimal = dc1 if mse1 < min(mse2, mse3) else dc2 if mse2 < min(mse1, mse3) else dc3
    return dc_optimal


def smod_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    count1, count2, count3 = 0, 0, 0
    diff1, diff2, diff3 = 0., 0., 0.
    if dct_left is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_left[0:8, 7] - spatial_left[0:8, 6] + 2*(spatial_left[0:8, 7] - spatial_target[0:8, 0]) + spatial_target[0:8, 1] - spatial_target[0:8, 0]
        diff2 += spatial_left[1:8, 7] - spatial_left[0:7, 6] + 2*(spatial_left[0:7, 7] - spatial_target[1:8, 0]) + spatial_target[1:8, 1] - spatial_target[0:7, 0]
        diff3 += spatial_left[0:7, 7] - spatial_left[1:8, 6] + 2*(spatial_left[1:8, 7] - spatial_target[0:7, 0]) + spatial_target[0:7, 1] - spatial_target[1:8, 0]

    if dct_up is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_up[7, 0:8] - spatial_up[6, 0:8] + 2*(spatial_up[7, 0:8] - spatial_target[0, 0:8]) + spatial_target[1, 0:8] - spatial_target[0, 0:8]
        diff2 += spatial_up[7, 1:8] - spatial_up[6, 0:7] + 2*(spatial_up[7, 0:7] - spatial_target[0, 1:8]) + spatial_target[1, 1:8] - spatial_target[0, 0:7]
        diff3 += spatial_up[7, 0:7] - spatial_up[6, 1:8] + 2*(spatial_up[7, 1:8] - spatial_target[0, 0:7]) + spatial_target[1, 0:7] - spatial_target[0, 1:8]

    if dct_right is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_right[0:8, 0] - spatial_right[0:8, 1] + 2*(spatial_right[0:8, 0] - spatial_target[0:8, 7]) + spatial_target[0:8, 6] - spatial_target[0:8, 7]
        diff2 += spatial_right[1:8, 0] - spatial_right[0:7, 1] + 2*(spatial_right[0:7, 0] - spatial_target[1:8, 7]) + spatial_target[1:8, 6] - spatial_target[0:7, 7]
        diff3 += spatial_right[0:7, 0] - spatial_right[1:8, 1] + 2*(spatial_right[1:8, 0] - spatial_target[0:7, 7]) + spatial_target[0:7, 6] - spatial_target[1:8, 7]

    if dct_down is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_down[0, 0:8] - spatial_down[1, 0:8] + 2*(spatial_down[0, 0:8] - spatial_target[7, 0:8]) + spatial_target[6, 0:8] - spatial_target[7, 0:8]
        diff2 += spatial_down[0, 1:8] - spatial_down[1, 0:7] + 2*(spatial_down[0, 0:7] - spatial_target[7, 1:8]) + spatial_target[6, 1:8] - spatial_target[7, 0:7]
        diff3 += spatial_down[0, 0:7] - spatial_down[1, 1:8] + 2*(spatial_down[0, 1:8] - spatial_target[7, 0:7]) + spatial_target[6, 0:7] - spatial_target[7, 1:8]

    dc1 = np.clip(8 * np.sum(diff1) / (count1 * Q[0][0]), DC_border[0], DC_border[1])
    dc2 = np.clip(8 * np.sum(diff2) / (count2 * Q[0][0]), DC_border[0], DC_border[1])
    dc3 = np.clip(8 * np.sum(diff3) / (count3 * Q[0][0]), DC_border[0], DC_border[1])
    # dc1 = np.clip(np.round(8 * np.sum(diff1) / (count1 * Q[0][0])), DC_border[0], DC_border[1])
    # dc2 = np.clip(np.round(8 * np.sum(diff2) / (count2 * Q[0][0])), DC_border[0], DC_border[1])
    # dc3 = np.clip(np.round(8 * np.sum(diff3) / (count3 * Q[0][0])), DC_border[0], DC_border[1])

    mse1 = 0.
    mse2 = 0.
    mse3 = 0.
    dct_target[0, 0] = dc1
    spatial_target1 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc2
    spatial_target2 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc3
    spatial_target3 = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1 += np.sum(np.square(spatial_left[0:8, 7] - spatial_left[0:8, 6] + 2*(spatial_left[0:8, 7] - spatial_target1[0:8, 0]) + spatial_target1[0:8, 1] - spatial_target1[0:8, 0])) / 8
        mse2 += np.sum(np.square(spatial_left[1:8, 7] - spatial_left[0:7, 6] + 2*(spatial_left[0:7, 7] - spatial_target2[1:8, 0]) + spatial_target2[1:8, 1] - spatial_target2[0:7, 0])) / 7
        mse3 += np.sum(np.square(spatial_left[0:7, 7] - spatial_left[1:8, 6] + 2*(spatial_left[1:8, 7] - spatial_target3[0:7, 0]) + spatial_target3[0:7, 1] - spatial_target3[1:8, 0])) / 7
    if dct_up is not False:
        mse1 += np.sum(np.square(spatial_up[7, 0:8] - spatial_up[6, 0:8] + 2*(spatial_up[7, 0:8] - spatial_target1[0, 0:8]) + spatial_target1[1, 0:8] - spatial_target1[0, 0:8])) / 8
        mse2 += np.sum(np.square(spatial_up[7, 1:8] - spatial_up[6, 0:7] + 2*(spatial_up[7, 0:7] - spatial_target2[0, 1:8]) + spatial_target2[1, 1:8] - spatial_target2[0, 0:7])) / 7
        mse3 += np.sum(np.square(spatial_up[7, 0:7] - spatial_up[6, 1:8] + 2*(spatial_up[7, 1:8] - spatial_target3[0, 0:7]) + spatial_target3[1, 0:7] - spatial_target3[0, 1:8])) / 7
    if dct_right is not False:
        mse1 += np.sum(np.square(spatial_right[0:8, 0] - spatial_right[0:8, 1] + 2*(spatial_right[0:8, 0] - spatial_target1[0:8, 7]) + spatial_target1[0:8, 6] - spatial_target1[0:8, 7])) / 8
        mse2 += np.sum(np.square(spatial_right[1:8, 0] - spatial_right[0:7, 1] + 2*(spatial_right[0:7, 0] - spatial_target2[1:8, 7]) + spatial_target2[1:8, 6] - spatial_target2[0:7, 7])) / 7
        mse3 += np.sum(np.square(spatial_right[0:7, 0] - spatial_right[1:8, 1] + 2*(spatial_right[1:8, 0] - spatial_target3[0:7, 7]) + spatial_target3[0:7, 6] - spatial_target3[1:8, 7])) / 7
    if dct_down is not False:
        mse1 += np.sum(np.square(spatial_down[0, 0:8] - spatial_down[1, 0:8] + 2*(spatial_down[0, 0:8] - spatial_target1[7, 0:8]) + spatial_target1[6, 0:8] - spatial_target1[7, 0:8])) / 8
        mse2 += np.sum(np.square(spatial_down[0, 1:8] - spatial_down[1, 0:7] + 2*(spatial_down[0, 0:7] - spatial_target2[7, 1:8]) + spatial_target2[6, 1:8] - spatial_target2[7, 0:7])) / 7
        mse3 += np.sum(np.square(spatial_down[0, 0:7] - spatial_down[1, 1:8] + 2*(spatial_down[0, 1:8] - spatial_target3[7, 0:7]) + spatial_target3[6, 0:7] - spatial_target3[7, 1:8])) / 7
    
    dc_optimal = dc1 if mse1 < min(mse2, mse3) else dc2 if mse2 < min(mse1, mse3) else dc3
    return dc_optimal


def mmmod_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    count1, count2, count3 = 0, 0, 0
    diff1, diff2, diff3 = 0., 0., 0.
    if dct_left is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_left[0:8, 5] + spatial_left[0:8, 7] + spatial_left[0:8, 7] + spatial_target[0:8, 1] + spatial_target[0:8, 1] - spatial_left[0:8, 6] - spatial_left[0:8, 6] - spatial_target[0:8, 0] - spatial_target[0:8, 0] - spatial_target[0:8, 2]
        diff2 += spatial_left[0:7, 5] + spatial_left[0:7, 7] + spatial_left[1:8, 7] + spatial_target[1:8, 1] + spatial_target[0:7, 1] - spatial_left[1:8, 6] - spatial_left[0:7, 6] - spatial_target[1:8, 0] - spatial_target[0:7, 0] - spatial_target[1:8, 2]
        diff3 += spatial_left[1:8, 5] + spatial_left[1:8, 7] + spatial_left[0:7, 7] + spatial_target[0:7, 1] + spatial_target[1:8, 1] - spatial_left[0:7, 6] - spatial_left[1:8, 6] - spatial_target[0:7, 0] - spatial_target[1:8, 0] - spatial_target[0:7, 2]

    if dct_up is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_up[5, 0:8] + spatial_up[7, 0:8] + spatial_up[7, 0:8] + spatial_target[1, 0:8] + spatial_target[1, 0:8] - spatial_up[6, 0:8] - spatial_up[6, 0:8] - spatial_target[0, 0:8] - spatial_target[0, 0:8] - spatial_target[2, 0:8]
        diff2 += spatial_up[5, 0:7] + spatial_up[7, 0:7] + spatial_up[7, 1:8] + spatial_target[1, 1:8] + spatial_target[1, 0:7] - spatial_up[6, 1:8] - spatial_up[6, 0:7] - spatial_target[0, 1:8] - spatial_target[0, 0:7] - spatial_target[2, 1:8]
        diff3 += spatial_up[5, 1:8] + spatial_up[7, 1:8] + spatial_up[7, 0:7] + spatial_target[1, 0:7] + spatial_target[1, 1:8] - spatial_up[6, 0:7] - spatial_up[6, 1:8] - spatial_target[0, 0:7] - spatial_target[0, 1:8] - spatial_target[2, 0:7]

    if dct_right is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_right[0:8, 2] + spatial_right[0:8, 0] + spatial_right[0:8, 0] + spatial_target[0:8, 6] + spatial_target[0:8, 6] - spatial_right[0:8, 1] - spatial_right[0:8, 1] - spatial_target[0:8, 7] - spatial_target[0:8, 7] - spatial_target[0:8, 5]
        diff2 += spatial_right[0:7, 2] + spatial_right[0:7, 0] + spatial_right[1:8, 0] + spatial_target[1:8, 6] + spatial_target[0:7, 6] - spatial_right[1:8, 1] - spatial_right[0:7, 1] - spatial_target[1:8, 7] - spatial_target[0:7, 7] - spatial_target[1:8, 5]
        diff3 += spatial_right[1:8, 2] + spatial_right[1:8, 0] + spatial_right[0:7, 0] + spatial_target[0:7, 6] + spatial_target[1:8, 6] - spatial_right[0:7, 1] - spatial_right[1:8, 1] - spatial_target[0:7, 7] - spatial_target[1:8, 7] - spatial_target[0:7, 5]

    if dct_down is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += spatial_down[2, 0:8] + spatial_down[0, 0:8] + spatial_down[0, 0:8] + spatial_target[6, 0:8] + spatial_target[6, 0:8] - spatial_down[1, 0:8] - spatial_down[1, 0:8] - spatial_target[7, 0:8] - spatial_target[7, 0:8] - spatial_target[5, 0:8]
        diff2 += spatial_down[2, 0:7] + spatial_down[0, 0:7] + spatial_down[0, 1:8] + spatial_target[6, 1:8] + spatial_target[6, 0:7] - spatial_down[1, 1:8] - spatial_down[1, 0:7] - spatial_target[7, 1:8] - spatial_target[7, 0:7] - spatial_target[5, 1:8]
        diff3 += spatial_down[2, 1:8] + spatial_down[0, 1:8] + spatial_down[0, 0:7] + spatial_target[6, 0:7] + spatial_target[6, 1:8] - spatial_down[1, 0:7] - spatial_down[1, 1:8] - spatial_target[7, 0:7] - spatial_target[7, 1:8] - spatial_target[5, 0:7]

    dc1 = np.clip(8 * np.sum(diff1) / (count1 * Q[0][0]), DC_border[0], DC_border[1])
    dc2 = np.clip(8 * np.sum(diff2) / (count2 * Q[0][0]), DC_border[0], DC_border[1])
    dc3 = np.clip(8 * np.sum(diff3) / (count3 * Q[0][0]), DC_border[0], DC_border[1])
    # dc1 = np.clip(np.round(8 * np.sum(diff1) / (count1 * Q[0][0])), DC_border[0], DC_border[1])
    # dc2 = np.clip(np.round(8 * np.sum(diff2) / (count2 * Q[0][0])), DC_border[0], DC_border[1])
    # dc3 = np.clip(np.round(8 * np.sum(diff3) / (count3 * Q[0][0])), DC_border[0], DC_border[1])

    mse1 = 0.
    mse2 = 0.
    mse3 = 0.
    dct_target[0, 0] = dc1
    spatial_target1 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc2
    spatial_target2 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc3
    spatial_target3 = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1 += np.sum(np.square((spatial_left[0:8, 7] - spatial_target1[0:8, 0]) -
                                    (spatial_left[0:8, 6] - spatial_left[0:8, 7]))) / 8
        mse2 += np.sum(np.square((spatial_left[0:7, 7] - spatial_target2[1:8, 0]) -
                                    (spatial_left[0:7, 6] - spatial_left[1:8, 7]))) / 7
        mse3 += np.sum(np.square((spatial_left[1:8, 7] - spatial_target3[0:7, 0]) -
                                    (spatial_left[1:8, 6] - spatial_left[0:7, 7]))) / 7
    if dct_up is not False:
        mse1 += np.sum(np.square((spatial_up[7, 0:8] - spatial_target1[0, 0:8]) -
                                    (spatial_up[6, 0:8] - spatial_up[7, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_up[7, 0:7] - spatial_target2[0, 1:8]) -
                                    (spatial_up[6, 0:7] - spatial_up[7, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_up[7, 1:8] - spatial_target3[0, 0:7]) -
                                    (spatial_up[6, 1:8] - spatial_up[7, 0:7]))) / 7
    if dct_right is not False:
        mse1 += np.sum(np.square((spatial_right[0:8, 0] - spatial_target1[0:8, 7]) -
                                    (spatial_right[0:8, 1] - spatial_right[0:8, 0]))) / 8
        mse2 += np.sum(np.square((spatial_right[0:7, 0] - spatial_target2[1:8, 7]) -
                                    (spatial_right[0:7, 1] - spatial_right[1:8, 0]))) / 7
        mse3 += np.sum(np.square((spatial_right[1:8, 0] - spatial_target3[0:7, 7]) -
                                    (spatial_right[1:8, 1] - spatial_right[0:7, 0]))) / 7
    if dct_down is not False:
        mse1 += np.sum(np.square((spatial_down[0, 0:8] - spatial_target1[7, 0:8]) -
                                    (spatial_down[1, 0:8] - spatial_down[0, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_down[0, 0:7] - spatial_target2[7, 1:8]) -
                                    (spatial_down[1, 0:7] - spatial_down[0, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_down[0, 1:8] - spatial_target3[7, 0:7]) -
                                    (spatial_down[1, 1:8] - spatial_down[0, 0:7]))) / 7
    
    dc_optimal = dc1 if mse1 < min(mse2, mse3) else dc2 if mse2 < min(mse1, mse3) else dc3
    return dc_optimal


def mmmmod_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    count1, count2, count3 = 0, 0, 0
    diff1, diff2, diff3 = 0., 0., 0.
    if dct_left is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_target[0:8, 0] - spatial_target[0:8, 1])
        diff2 += (spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_target[0:7, 0] - spatial_target[1:8, 1])
        diff3 += (spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_target[1:8, 0] - spatial_target[0:7, 1])

    if dct_up is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_target[0, 0:8] - spatial_target[1, 0:8])
        diff2 += (spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_target[0, 0:7] - spatial_target[1, 1:8])
        diff3 += (spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_target[0, 1:8] - spatial_target[1, 0:7])

    if dct_right is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_target[0:8, 7] - spatial_target[0:8, 6])
        diff2 += (spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_target[0:7, 7] - spatial_target[1:8, 6])
        diff3 += (spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_target[1:8, 7] - spatial_target[0:7, 6])

    if dct_down is not False:
        count1 += 8
        count2 += 7
        count3 += 7
        diff1 += (spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_target[7, 0:8] - spatial_target[6, 0:8])
        diff2 += (spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_target[7, 0:7] - spatial_target[6, 1:8])
        diff3 += (spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_target[7, 1:8] - spatial_target[6, 0:7])

    dc1 = np.clip(8 * np.sum(diff1) / (count1 * Q[0][0]), DC_border[0], DC_border[1])
    dc2 = np.clip(8 * np.sum(diff2) / (count2 * Q[0][0]), DC_border[0], DC_border[1])
    dc3 = np.clip(8 * np.sum(diff3) / (count3 * Q[0][0]), DC_border[0], DC_border[1])
    # dc1 = np.clip(np.round(8 * np.sum(diff1) / (count1 * Q[0][0])), DC_border[0], DC_border[1])
    # dc2 = np.clip(np.round(8 * np.sum(diff2) / (count2 * Q[0][0])), DC_border[0], DC_border[1])
    # dc3 = np.clip(np.round(8 * np.sum(diff3) / (count3 * Q[0][0])), DC_border[0], DC_border[1])

    mse1 = 0.
    mse2 = 0.
    mse3 = 0.
    dct_target[0, 0] = dc1
    spatial_target1 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc2
    spatial_target2 = cv2.idct(dct_target * Q) + 128
    dct_target[0, 0] = dc3
    spatial_target3 = cv2.idct(dct_target * Q) + 128
    if dct_left is not False:
        mse1 += np.sum(np.square((spatial_left[0:8, 7] - spatial_target1[0:8, 0]) - (spatial_target1[0:8, 0] - spatial_target1[0:8, 1]))) / 8
        mse2 += np.sum(np.square((spatial_left[0:7, 7] - spatial_target2[1:8, 0]) - (spatial_target2[0:7, 0] - spatial_target2[1:8, 1]))) / 7
        mse3 += np.sum(np.square((spatial_left[1:8, 7] - spatial_target3[0:7, 0]) - (spatial_target3[1:8, 0] - spatial_target3[0:7, 1]))) / 7
    if dct_up is not False:
        mse1 += np.sum(np.square((spatial_up[7, 0:8] - spatial_target1[0, 0:8]) - (spatial_target1[0, 0:8] - spatial_target1[1, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_up[7, 0:7] - spatial_target2[0, 1:8]) - (spatial_target2[0, 0:7] - spatial_target2[1, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_up[7, 1:8] - spatial_target3[0, 0:7]) - (spatial_target3[0, 1:8] - spatial_target3[1, 0:7]))) / 7
    if dct_right is not False:
        mse1 += np.sum(np.square((spatial_right[0:8, 0] - spatial_target1[0:8, 7]) - (spatial_target1[0:8, 7] - spatial_target1[0:8, 6]))) / 8
        mse2 += np.sum(np.square((spatial_right[0:7, 0] - spatial_target2[1:8, 7]) - (spatial_target2[0:7, 7] - spatial_target2[1:8, 6]))) / 7
        mse3 += np.sum(np.square((spatial_right[1:8, 0] - spatial_target3[0:7, 7]) - (spatial_target3[1:8, 7] - spatial_target3[0:7, 6]))) / 7
    if dct_down is not False:
        mse1 += np.sum(np.square((spatial_down[0, 0:8] - spatial_target1[7, 0:8]) - (spatial_target1[7, 0:8] - spatial_target1[6, 0:8]))) / 8
        mse2 += np.sum(np.square((spatial_down[0, 0:7] - spatial_target2[7, 1:8]) - (spatial_target2[7, 0:7] - spatial_target2[6, 1:8]))) / 7
        mse3 += np.sum(np.square((spatial_down[0, 1:8] - spatial_target3[7, 0:7]) - (spatial_target3[7, 1:8] - spatial_target3[6, 0:7]))) / 7
    
    dc_optimal = dc1 if mse1 < min(mse2, mse3) else dc2 if mse2 < min(mse1, mse3) else dc3
    return dc_optimal


def log_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    def dc_opt(x):
        cnt = 0
        diff = 0.
        if dct_left is not False:
            cnt += 1
            diff += np.sum(np.log((x + spatial_target[0:8, 0] - spatial_left[0:8, 7])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[1:8, 0] - spatial_left[0:7, 7])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[0:7, 0] - spatial_left[1:8, 7])**2 + 1))
        if dct_right is not False:
            cnt += 1
            diff += np.sum(np.log((x + spatial_target[0:8, 7] - spatial_right[0:8, 0])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[1:8, 7] - spatial_right[0:7, 0])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[0:7, 7] - spatial_right[1:8, 0])**2 + 1))
        if dct_up is not False:
            cnt += 1
            diff += np.sum(np.log((x + spatial_target[0, 0:8] - spatial_up[7, 0:8])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[0, 1:8] - spatial_up[7, 0:7])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[0, 0:7] - spatial_up[7, 1:8])**2 + 1))
        if dct_down is not False:
            cnt += 1
            diff += np.sum(np.log((x + spatial_target[7, 0:8] - spatial_down[0, 0:8])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[7, 1:8] - spatial_down[0, 0:7])**2 + 1)) \
                  + np.sum(np.log((x + spatial_target[7, 0:7] - spatial_down[0, 1:8])**2 + 1))
        
        return diff / (22 * cnt)

    def dc_con():
        con = ({'type': 'ineq', 'fun': lambda x: x + 128}, \
               {'type': 'ineq', 'fun': lambda x: -x + 127})
        return con

    x0 = 0
    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    res = minimize(dc_opt, x0, method='SLSQP', constraints=dc_con())
    return np.clip(8 * res.x / Q[0][0], DC_border[0], DC_border[1])


def avg_estimate(Q, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    dct_target[0, 0] = 0
    spatial_target = cv2.idct(dct_target * Q) + 128

    cnt = 0
    cad = []
    if dct_left is not False:
        cnt += 1
        cad += ((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_left[0:8, 6] - spatial_left[0:8, 7])).tolist()
        cad += ((spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_left[0:7, 6] - spatial_left[1:8, 7])).tolist()
        cad += ((spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_left[1:8, 6] - spatial_left[0:7, 7])).tolist()
        cad += ((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_target[0:8, 0] - spatial_target[0:8, 1])).tolist()
        cad += ((spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_target[0:7, 0] - spatial_target[1:8, 1])).tolist()
        cad += ((spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_target[1:8, 0] - spatial_target[0:7, 1])).tolist()

    if dct_right is not False:
        cnt += 1
        cad += ((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_right[0:8, 1] - spatial_right[0:8, 0])).tolist()
        cad += ((spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_right[0:7, 1] - spatial_right[1:8, 0])).tolist()
        cad += ((spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_right[1:8, 1] - spatial_right[0:7, 0])).tolist()
        cad += ((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_target[0:8, 7] - spatial_target[0:8, 6])).tolist()
        cad += ((spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_target[0:7, 7] - spatial_target[1:8, 6])).tolist()
        cad += ((spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_target[1:8, 7] - spatial_target[0:7, 6])).tolist()

    if dct_up is not False:
        cnt += 1
        cad += ((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_up[6, 0:8] - spatial_up[7, 0:8])).tolist()
        cad += ((spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_up[6, 0:7] - spatial_up[7, 1:8])).tolist()
        cad += ((spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_up[6, 1:8] - spatial_up[7, 0:7])).tolist()
        cad += ((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_target[0, 0:8] - spatial_target[1, 0:8])).tolist()
        cad += ((spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_target[0, 0:7] - spatial_target[1, 1:8])).tolist()
        cad += ((spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_target[0, 1:8] - spatial_target[1, 0:7])).tolist()

    if dct_down is not False:
        cnt += 1
        cad += ((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_down[1, 0:8] - spatial_down[0, 0:8])).tolist()
        cad += ((spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_down[1, 0:7] - spatial_down[0, 1:8])).tolist()
        cad += ((spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_down[1, 1:8] - spatial_down[0, 0:7])).tolist()
        cad += ((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_target[7, 0:8] - spatial_target[6, 0:8])).tolist()
        cad += ((spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_target[7, 0:7] - spatial_target[6, 1:8])).tolist()
        cad += ((spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_target[7, 1:8] - spatial_target[6, 0:7])).tolist()

    cnt *= 16
    cad = sorted(cad)
    length = len(cad) - cnt
    diff = cad[length//2: -((length+1)//2)]
    avg_diff = sum(diff) / len(diff)
    DC_border = np.array([-1024, 1024]) / Q[0, 0]
    dc_optimal = np.clip(8 * avg_diff / Q[0, 0], DC_border[0], DC_border[1])
    return dc_optimal


def avg_spd_estimate(spatial_target, spatial_left=False, spatial_up=False, spatial_right=False, spatial_down=False):
    """
    Estimation of image block with the help of adjacent blocks.
    :param spatial_target: ndarray, image of target block
    :param spatial_left: ndarray, image of left block
    :param spatial_up: ndarray, image of upper block
    :param spatial_right: ndarray, image of right block
    :param spatial_down: ndarray, image of nether block
    :return: spatial_rec: float, optimal recovery of image
    """
    cnt = 0
    cad = []
    if spatial_left is not False:
        cnt += 1
        cad += ((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_left[0:8, 6] - spatial_left[0:8, 7])).tolist()
        cad += ((spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_left[0:7, 6] - spatial_left[1:8, 7])).tolist()
        cad += ((spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_left[1:8, 6] - spatial_left[0:7, 7])).tolist()
        cad += ((spatial_left[0:8, 7] - spatial_target[0:8, 0]) - (spatial_target[0:8, 0] - spatial_target[0:8, 1])).tolist()
        cad += ((spatial_left[0:7, 7] - spatial_target[1:8, 0]) - (spatial_target[0:7, 0] - spatial_target[1:8, 1])).tolist()
        cad += ((spatial_left[1:8, 7] - spatial_target[0:7, 0]) - (spatial_target[1:8, 0] - spatial_target[0:7, 1])).tolist()

    if spatial_right is not False:
        cnt += 1
        cad += ((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_right[0:8, 1] - spatial_right[0:8, 0])).tolist()
        cad += ((spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_right[0:7, 1] - spatial_right[1:8, 0])).tolist()
        cad += ((spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_right[1:8, 1] - spatial_right[0:7, 0])).tolist()
        cad += ((spatial_right[0:8, 0] - spatial_target[0:8, 7]) - (spatial_target[0:8, 7] - spatial_target[0:8, 6])).tolist()
        cad += ((spatial_right[0:7, 0] - spatial_target[1:8, 7]) - (spatial_target[0:7, 7] - spatial_target[1:8, 6])).tolist()
        cad += ((spatial_right[1:8, 0] - spatial_target[0:7, 7]) - (spatial_target[1:8, 7] - spatial_target[0:7, 6])).tolist()

    if spatial_up is not False:
        cnt += 1
        cad += ((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_up[6, 0:8] - spatial_up[7, 0:8])).tolist()
        cad += ((spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_up[6, 0:7] - spatial_up[7, 1:8])).tolist()
        cad += ((spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_up[6, 1:8] - spatial_up[7, 0:7])).tolist()
        cad += ((spatial_up[7, 0:8] - spatial_target[0, 0:8]) - (spatial_target[0, 0:8] - spatial_target[1, 0:8])).tolist()
        cad += ((spatial_up[7, 0:7] - spatial_target[0, 1:8]) - (spatial_target[0, 0:7] - spatial_target[1, 1:8])).tolist()
        cad += ((spatial_up[7, 1:8] - spatial_target[0, 0:7]) - (spatial_target[0, 1:8] - spatial_target[1, 0:7])).tolist()

    if spatial_down is not False:
        cnt += 1
        cad += ((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_down[1, 0:8] - spatial_down[0, 0:8])).tolist()
        cad += ((spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_down[1, 0:7] - spatial_down[0, 1:8])).tolist()
        cad += ((spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_down[1, 1:8] - spatial_down[0, 0:7])).tolist()
        cad += ((spatial_down[0, 0:8] - spatial_target[7, 0:8]) - (spatial_target[7, 0:8] - spatial_target[6, 0:8])).tolist()
        cad += ((spatial_down[0, 0:7] - spatial_target[7, 1:8]) - (spatial_target[7, 0:7] - spatial_target[6, 1:8])).tolist()
        cad += ((spatial_down[0, 1:8] - spatial_target[7, 0:7]) - (spatial_target[7, 1:8] - spatial_target[6, 0:7])).tolist()

    cnt *= 16
    cad = sorted(cad)
    length = len(cad) - cnt
    diff = cad[length//2: -((length+1)//2)]
    avg_diff = sum(diff) / len(diff)
    spatial_rec = spatial_target + np.clip(avg_diff, -128, 128)
    return np.clip(spatial_rec, 0, 255)


def estimate(Qs, method, dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    dc_optimal = np.zeros(dct_target.shape[-1], dtype=np.float64)
    for i in range(dct_target.shape[-1]):
        target = dct_target[:, :, i]
        left = dct_left[:, :, i] if dct_left is not False else False
        up = dct_up[:, :, i] if dct_up is not False else False
        right = dct_right[:, :, i] if dct_right is not False else False
        down = dct_down[:, :, i] if dct_down is not False else False
        dc_optimal[i] = method(Qs[i], target, left, up, right, down)
    return dc_optimal

methods = {
    "org": org_estimate,
    "mod": mod_estimate,
    "mod_spd": mod_spd_estimate,
    "ehc": ehc_estimate,
    "avg": avg_estimate,
    "avg_spd": avg_spd_estimate,
}