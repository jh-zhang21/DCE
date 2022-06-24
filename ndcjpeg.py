"""
DC estimation for JPEG without DC coefficient.
Author: Jianghui Zhang
Email: jh-zhang21@mails.tsinghua.edu.cn
"""

import os
import cv2
import time
import psutil
p = psutil.Process()
print(p.cpu_affinity())
import logging
import argparse
import multiprocessing
import numpy as np
# np.set_printoptions(threshold=1000000)
from io import BytesIO
import utils
import estimation
from prediction import border_prediction, inside_prediction, decompress


image_suffix = [".png", ".jpg", ".jpeg"]

def logging_result(value):
    if value:
        print("Error: ", value)


def org_png(image, path=None, grayscale=False):
    if grayscale:
        image = image.astype(np.uint8)
    else:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YUV2BGR)
    
    if args.output and path:
        cv2.imwrite(path, image)
    return image


def std_jpg(image, path=None, grayscale=False):
    # image_rec = utils.jpeg(image, Qs)
    image_rec = image
    if grayscale:
        image = image_rec.astype(np.uint8)
    else:
        image = cv2.cvtColor(image_rec.astype(np.uint8), cv2.COLOR_YUV2BGR)
    if args.output and path:
        cv2.imwrite(path, image.astype(np.uint8))
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    else:
        _, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        io_buf = BytesIO(buf)
        image = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    return image


def ehc_jpg(image, path=None, grayscale=False):
    image = image.astype(np.float64)
    h_n, w_n = image.shape[0] // 8, image.shape[1] // 8
    for i in range(h_n):
        for j in range(w_n):
            patch = image[i*8:(i+1)*8, j*8:(j+1)*8]
            patch -= patch.sum() / (8*8)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YUV2BGR)
    if args.output and path:
        cv2.imwrite(path, image.astype(np.uint8))
    return image


def rec_jpg(image, path=None, grayscale=False):
    image, _, inf_time = dc_recovery(image)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YUV2BGR)
    if args.output and path:
        cv2.imwrite(path, image)
    return image, inf_time


def rec_ehc_jpg(image, rec_path=None, ehc_path=None, grayscale=False):
    rec_image, ehc_image, inf_time = dc_recovery(image)
    if grayscale:
        ehc_image = ehc_image.astype(np.uint8)
        rec_image = rec_image.astype(np.uint8)
    else:
        ehc_image = cv2.cvtColor(ehc_image.astype(np.uint8), cv2.COLOR_YUV2BGR)
        rec_image = cv2.cvtColor(rec_image.astype(np.uint8), cv2.COLOR_YUV2BGR)
    if args.output:
        if rec_path:
            cv2.imwrite(rec_path, rec_image)
        if ehc_path:
            cv2.imwrite(ehc_path, ehc_image)
    return rec_image, ehc_image, inf_time


def write_image(image, org_pth=None, std_pth=None, ehc_pth=None, rec_pth=None, grayscale=False):
    """
    Convert image from BGR to YUV and save it.
    :param image: ndarray, YUV image
    :param image_path: string, path of image file
    """
    org_image = org_png(image, org_pth, grayscale)
    std_image = std_jpg(image, std_pth, grayscale)
    rec_image, ehc_image, inf_time = rec_ehc_jpg(image, rec_pth, ehc_pth, grayscale)
    return org_image, std_image, rec_image, ehc_image, inf_time


def dct_transform(image, mode='compress', dc_free=False):
    """
    Calculate DCT coefficients of an image.
    :param image: ndarray, input image
    :param mode: string, compression or normal mode
    :param dc_free: boolean, option whether DC coefficient is missing
    :return: dct_coefs
    """
    image = image - 128
    h, w = image.shape[0], image.shape[1]
    h_n, w_n = h // 8, w // 8
    c_n = image.shape[2] if image.ndim == 3 else 1
    image = image.reshape([h, w, c_n])
    dct_coefs = np.zeros([h_n, w_n, 8, 8, c_n], dtype=np.float64)
    for k in range(c_n):
        for i in range(h_n):
            for j in range(w_n):
                image_patch = image[i*8:(i+1)*8, j*8:(j+1)*8]
                dct = cv2.dct(image_patch[:, :, k]).astype(np.float64)

                if dc_free:
                    dct[0, 0] = 0.
                if mode == 'compress':
                    dct_coefs[i, j, :, :, k] = np.round(dct / Qs[k])
                elif mode == 'normal':
                    dct_coefs[i, j] = dct

    return dct_coefs


def idct_transform(dct_coefs, mode='compress'):
    """
    Recover image from DCT coefficients.
    :param dct_coefs: ndarray, DCT coefficients of blocks
    :param mode: string, compression or normal mode
    :return: image_rec: ndarray, recovered image
    """
    h_n, w_n, c_n = dct_coefs.shape[0], dct_coefs.shape[1], dct_coefs.shape[4]
    h = dct_coefs.shape[0] * dct_coefs.shape[2]
    w = dct_coefs.shape[1] * dct_coefs.shape[3]
    image_rec = np.zeros([h, w, c_n], dtype=np.float64)
    if mode == 'normal':
        for i in range(h_n):
            for j in range(w_n):
                idct = np.zeros_like(dct_coefs[i, j], dtype=np.float64)
                for k in range(c_n):
                    idct[:, :, k] = cv2.idct(dct_coefs[i, j, :, :, k])
                image_rec[i*8:(i+1)*8, j*8:(j+1)*8] = idct + 128
    
    elif mode == 'compress':
        for i in range(h_n):
            for j in range(w_n):
                idct = np.zeros_like(dct_coefs[i, j], dtype=np.float64)
                for k in range(c_n):
                    idct[:, :, k] = cv2.idct(dct_coefs[i, j, :, :, k] * Qs[k])
                image_rec[i*8:(i+1)*8, j*8:(j+1)*8] = idct + 128

    image_rec = np.round(image_rec.squeeze())
    return np.clip(image_rec, 0.0, 255.0)
    

def dc_recovery(image):
    """
    Recover image from four corners.
    :param: image: ndarray, original image
    :return: dct_preds: ndarray, predicted DCT coefficients of blocks
             image_rec: ndarray, recovered image
    """
    dct_coefs = dct_transform(image, mode='compress', dc_free=True)
    dct_coefs_corner = dct_transform(image, mode='compress', dc_free=False)
    h_n, w_n = dct_coefs.shape[0], dct_coefs.shape[1]
    # ehc
    dct_free = dct_coefs.copy()
    dct_free[0, 0, 0, 0] = dct_coefs_corner[0, 0, 0, 0]
    dct_free[0, w_n - 1, 0, 0] = dct_coefs_corner[0, w_n - 1, 0, 0]
    dct_free[h_n - 1, 0, 0, 0] = dct_coefs_corner[h_n - 1, 0, 0, 0]
    dct_free[h_n - 1, w_n - 1, 0, 0] = dct_coefs_corner[h_n - 1, w_n - 1, 0, 0]
    image_ehc = idct_transform(dct_free, mode='compress')
    # rec
    p.cpu_affinity([10])
    inf_time = time.time()

    ## original pipeline
    prediction = border_prediction(dct_coefs, dct_coefs_corner[:, :, 0, 0])
    dct_preds = prediction.prediction(Qs, estimation.methods[args.estimate])
    image_rec = idct_transform(dct_preds, mode='compress')

    # ## accelerated pipeline
    # decompressor = decompress(Qs, dct_coefs, dct_coefs_corner[:, :, 0, 0])
    # image_rec = decompressor.idct_transform()

    inf_time = time.time() - inf_time
    p.cpu_affinity([])

    return image_rec, image_ehc, inf_time


def get_args():
    argparser = argparse.ArgumentParser("DC recovery")
    argparser.add_argument('--log_path', type=str, default='./log/DCRecovery', help="Path to save log.")
    argparser.add_argument('--dataset_path', type=str, default='../DataSets/LFW', help="Image dataset path.")
    argparser.add_argument('--org_imgs_path', type=str, default='./dataset/LFW_org', help="Input image path.")
    argparser.add_argument('--std_imgs_path', type=str, default='./dataset/LFW_std', help="Output path of image compressed by standard JPEG.")
    argparser.add_argument('--ehc_imgs_path', type=str, default='./dataset/LFW_ehc', help="Output path of image compressed by enhanced JPEG.")
    argparser.add_argument('--rec_imgs_path', type=str, default='./dataset/LFW_rec', help="Output path of image recovered from enhanced JPEG.")
    argparser.add_argument('--weight', type=int, default=256, help="Image weight.")
    argparser.add_argument('--height', type=int, default=256, help="Image height.")
    argparser.add_argument('--quality', type=int, default=75, help="The JPEG image quality factor.")
    argparser.add_argument('--estimate', type=str, default='avg_spd', help="Which estimated method to use.")
    argparser.add_argument('--resize', default=False, action="store_true", help="Whether resize image.")
    argparser.add_argument('--grayscale', default=False, action="store_true", help="Grayscale or BGR.")
    argparser.add_argument('--output', default=False, action='store_true', help='Whether output image or not.')
    args = argparser.parse_args()

    # Check the path
    os.makedirs(args.log_path, exist_ok=True)
    if args.output:
        os.makedirs(args.org_imgs_path, exist_ok=True)
        os.makedirs(args.std_imgs_path, exist_ok=True)
        os.makedirs(args.ehc_imgs_path, exist_ok=True)
        os.makedirs(args.rec_imgs_path, exist_ok=True)
    
    # Check the estimated methods
    assert args.estimate in estimation.methods, "{} is not supported.".format(args.estimate)

    return args


def main(args):
    # DC Recovery
    count = 0
    inf_time_avg = 0.
    std_psnr_avg, std_ssim_avg = 0., 0.
    ehc_psnr_avg, ehc_ssim_avg = 0., 0.
    rec_psnr_avg, rec_ssim_avg = 0., 0.
    std_rec_psnr_avg, std_rec_ssim_avg = 0., 0.
    std_size_avg, ehc_size_avg, rec_size_avg = 0., 0., 0.

    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if os.path.splitext(file)[-1] in image_suffix:
                img_path = os.path.join(root, file)
                file_name = ".".join(os.path.splitext(file)[:-1])
                org_path = os.path.join(args.org_imgs_path, file_name + ".png")
                std_path = os.path.join(args.std_imgs_path, file_name + ".jpg")
                ehc_path = os.path.join(args.ehc_imgs_path, file_name + ".jpg")
                rec_path = os.path.join(args.rec_imgs_path, file_name + ".jpg")
                assert os.path.exists(img_path)
                image = utils.read_image(img_path, args.grayscale)
                if args.resize:
                    image = utils.crop_resize(image, (args.weight, args.height))
                org_image, std_image, rec_image, ehc_image, inf_time = write_image(image, org_path, std_path, ehc_path, rec_path, args.grayscale)
                h, w = rec_image.shape[0], rec_image.shape[1]
                org_image = org_image[:h, :w]
                std_image = std_image[:h, :w]
                assert args.output
                org_size = os.path.getsize(img_path)
                std_size = os.path.getsize(std_path) / org_size
                ehc_size = os.path.getsize(ehc_path) / org_size
                rec_size = os.path.getsize(rec_path) / org_size
                # std_psnr, std_ssim = peak_signal_noise_ratio(std_image, image, data_range=255), structural_similarity(std_image, image, data_range=255, multichannel=True)
                # ehc_psnr, ehc_ssim = peak_signal_noise_ratio(ehc_image, image, data_range=255), structural_similarity(ehc_image, image, data_range=255, multichannel=True)
                # rec_psnr, rec_ssim = peak_signal_noise_ratio(rec_image, image, data_range=255), structural_similarity(rec_image, image, data_range=255, multichannel=True)
                std_psnr, std_ssim = utils.calculate_psnr(std_image, org_image), utils.calculate_ssim(std_image, org_image)
                ehc_psnr, ehc_ssim = utils.calculate_psnr(ehc_image, org_image), utils.calculate_ssim(ehc_image, org_image)
                rec_psnr, rec_ssim = utils.calculate_psnr(rec_image, org_image), utils.calculate_ssim(rec_image, org_image)
                std_rec_psnr, std_rec_ssim = utils.calculate_psnr(std_image, rec_image), utils.calculate_ssim(std_image, rec_image)
                # std_psnr, std_ssim = tf.image.psnr(std_image, org_image, max_val=255).numpy().item(), tf.image.ssim(std_image, org_image, max_val=255).numpy().item()
                # ehc_psnr, ehc_ssim = tf.image.psnr(ehc_image, org_image, max_val=255).numpy().item(), tf.image.ssim(ehc_image, org_image, max_val=255).numpy().item()
                # rec_psnr, rec_ssim = tf.image.psnr(rec_image, org_image, max_val=255).numpy().item(), tf.image.ssim(rec_image, org_image, max_val=255).numpy().item()
                # std_rec_psnr, std_rec_ssim = tf.image.psnr(std_image, rec_image, max_val=255).numpy().item(), tf.image.ssim(std_image, rec_image, max_val=255).numpy().item()
                logger.info(" #### PSNR -> std: {:.4f}, ehc: {:.4f}, rec: {:.4f}, std-rec: {:.4f}".format(std_psnr, ehc_psnr, rec_psnr, std_rec_psnr))
                logger.info("      SSIM -> std: {:.4f}, ehc: {:.4f}, rec: {:.4f}, std-rec: {:.4f}".format(std_ssim, ehc_ssim, rec_ssim, std_rec_ssim))
                logger.info("      Size -> std: {:.4f}, ehc: {:.4f}, rec: {:.4f}".format(std_size, ehc_size, rec_size))
                logger.info("      Inf Time -> {:.4f}".format(inf_time))
                logger.info("      {}".format(img_path))
                count += 1
                std_size_avg += std_size
                ehc_size_avg += ehc_size
                rec_size_avg += rec_size
                std_psnr_avg += std_psnr
                std_ssim_avg += std_ssim
                ehc_psnr_avg += ehc_psnr
                ehc_ssim_avg += ehc_ssim
                rec_psnr_avg += rec_psnr
                rec_ssim_avg += rec_ssim
                std_rec_psnr_avg += std_rec_psnr
                std_rec_ssim_avg += std_rec_ssim
                inf_time_avg += inf_time

    inf_time_avg = inf_time_avg / count
    std_size_avg /= count
    ehc_size_avg /= count
    rec_size_avg /= count
    std_psnr_avg, std_ssim_avg = std_psnr_avg / count, std_ssim_avg / count
    ehc_psnr_avg, ehc_ssim_avg = ehc_psnr_avg / count, ehc_ssim_avg / count
    rec_psnr_avg, rec_ssim_avg = rec_psnr_avg / count, rec_ssim_avg / count
    std_rec_psnr_avg, std_rec_ssim_avg = std_rec_psnr_avg / count, std_rec_ssim_avg / count
    logger.info("######## DC Recovery end. ########")
    logger.info(" Avg PSNR -> std: {:.4f}, ehc: {:.4f}, rec: {:.4f}, std-rec: {:.4f}".format(std_psnr_avg, ehc_psnr_avg, rec_psnr_avg, std_rec_psnr_avg))
    logger.info(" Avg SSIM -> std: {:.4f}, ehc: {:.4f}, rec: {:.4f}, std-rec: {:.4f}".format(std_ssim_avg, ehc_ssim_avg, rec_ssim_avg, std_rec_ssim_avg))
    logger.info(" Avg Size -> std: {:.4f}, ehc: {:.4f}, rec: {:.4f}".format(std_size_avg, ehc_size_avg, rec_size_avg))
    logger.info(" Avg Inf Time -> {:.4f}".format(inf_time_avg))

    print("Dc_recovery finished.")


if __name__ == '__main__':
    # Parse the arguments
    args = get_args()
    
    # Quantization table
    Qs = utils.qtable(args.quality)

    # Logging info
    utils.setup_logger('base', args.log_path, "dc_recovery", screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info("######## DC Recovery start.")
    logger.info(args)

    # DC recovery
    main(args)
