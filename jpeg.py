import os
import cv2
import numpy as np


def dct(image, Qs):
    """
    Do DCT transform turn spatial domain to frequency domain, and quantize the DCT coefficients.
    """
    image = image - 128
    w, h = image.shape[0], image.shape[1]
    w_n, h_n = w // 8, h // 8
    c_n = image.shape[2] if image.ndim == 3 else 1
    image = image.reshape([w, h, c_n])
    coeff = np.zeros([w_n, h_n, 8, 8, c_n], dtype=np.float64)
    for k in range(c_n):
        for i in range(w_n):
            for j in range(h_n):
                image_patch = image[i*8:(i+1)*8, j*8:(j+1)*8]
                coeff_patch = cv2.dct(image_patch[:, :, k]).astype(np.float64)
                coeff[i, j, :, :, k] = np.round(coeff_patch / Qs[k])

    return coeff


def idct(coeff, Qs):
    """
    Inverse quantize the DCT coefficients, and do IDCT transform turn frequency domain to spatial domain.
    """
    w_n, h_n, c_n = coeff.shape[0], coeff.shape[1], coeff.shape[4]
    w = coeff.shape[0] * coeff.shape[2]
    h = coeff.shape[1] * coeff.shape[3]
    image_rec = np.zeros([w, h, c_n], dtype=np.float64)
    for i in range(w_n):
        for j in range(h_n):
            coeff_patch = np.zeros_like(coeff[i, j], dtype=np.float)
            for k in range(c_n):
                coeff_patch[:, :, k] = cv2.idct(coeff[i, j, :, :, k] * Qs[k])
            image_rec[i*8:(i+1)*8, j*8:(j+1)*8] = coeff_patch + 128

    image_rec = np.round(image_rec.squeeze()).clip(0.0, 255.0)

    return image_rec


def jpeg(image, Qs):
    """
    Standard JPEG algorithm without color transform and entropy coding.
    """
    coeff = dct(image, Qs)
    image_rec = idct(coeff, Qs)
    return image_rec
