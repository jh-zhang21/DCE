import os
import cv2
import math
import random
import logging
import numpy as np
from datetime import datetime


def get_timestamp():
    """
    Get current timestamp as string.
    """
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """
    Set up logger for printing info.
    """
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def crop_resize(image, shape):
    """
    Crop image in shape randomly.
    """
    iw, ih = image.shape[0] * 1., image.shape[1] * 1.
    iw, ih = iw / shape[0], ih / shape[1]
    factor = min(iw, ih)
    iw_n = int(shape[0] * factor + 0.5)
    ih_n = int(shape[1] * factor + 0.5)
    iw_i = random.randint(0, image.shape[0] - iw_n)
    ih_j = random.randint(0, image.shape[1] - ih_n)
    return cv2.resize(image[iw_i:iw_i+iw_n, ih_j:ih_j+ih_n], shape)


def read_image(image_path, grayscale=False):
    """
    Read an image and convert it from BGR(np.uint8) to YUV(np.float64).
    """
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float64)
    return image


def write_image(image, image_path, grayscale=False):
    """
    Write an image and convert it from YUV(np.float64) to BGR(np.uint8).
    """
    if grayscale:
        image = image.astype(np.uint8)
    else:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YUV2BGR)
    cv2.imwrite(image_path, image)


def calculate_psnr(image1, image2):
    """
    Calculate the psnr between two image.
    """
    assert image1.shape == image2.shape, 'Input images must have the same dimensions {} - {}.'.format(image1.shape, image2.shape)

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    mse = np.mean((image1 - image2)**2)
    if mse == 0:
        return np.float64(80)
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(image1, image2):
    """
    Calculate the ssim between two image.
    """
    ### calculate the ssim between same channel of two image. 
    def ssim_per_channel(img1_chl, img2_chl):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1_chl = img1_chl.astype(np.float64)
        img2_chl = img2_chl.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1_chl, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2_chl, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1_chl ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2_chl ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1_chl * img2_chl, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    assert image1.shape == image2.shape, 'Input images must have the same dimensions.'

    if image1.ndim == 2:
        return ssim_per_channel(image1, image2)
    elif image1.ndim == 3:
        if image1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_per_channel(image1, image2))
            return np.array(ssims).mean()
        elif image1.shape[2] == 1:
            return ssim_per_channel(np.squeeze(image1), np.squeeze(image2))
        else:
            raise ValueError('Wrong input image channels.')
    else:
        raise ValueError('Wrong input image dimensions.')


def qtable(quality):
    """
    Get quantization table specified by quality.
    """
    # Standard quantitation tables
    CyQ = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.int32)
    CbCrQ = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.int32)
    # Jpeg image quality factor
    q = quality
    assert 0 < q and q <= 100
    s = 5000 // q if q <= 50 else 200 - 2*q if q < 100 else 1
    # Dynamically adjust quantitation tables
    CyQ = (CyQ*s + 50) // 100
    CbCrQ = (CbCrQ*s + 50) // 100
    for i in range(CyQ.shape[0]):
        for j in range(CyQ.shape[1]):
            if CyQ[i, j] <= 0:
                CyQ[i, j] = 1

    for i in range(CbCrQ.shape[0]):
        for j in range(CbCrQ.shape[1]):
            if CbCrQ[i, j] <= 0:
                CbCrQ[i, j] = 1

    return np.array([CyQ, CbCrQ, CbCrQ], dtype=np.int32)

