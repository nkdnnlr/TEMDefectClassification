import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from skimage import exposure, filters


import os
import time

def get_variance_ratio(image, pc_0=0, pc_test=1, plot=False):
    """
    Moves window over images and collects Fourier transformations.
    Then calculates PCA and returns list of explained variance for all PCs
    :param image:
    :param pc_0:
    :param pc_test:
    :param plot:
    :return:
    """

    # Scale intensity of batch
    # p2, p98 = np.percentile(image, (2, 98))
    # image = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Slide moving window over image and save them
    height, width = image.shape
    window_size_y, window_size_x = [height // 8, width // 8]
    step_y, step_x = [window_size_y // 2, window_size_x // 2]
    windows = []
    for y in range(window_size_y+1, height, step_y):
        for x in range(window_size_x+1, width, step_x):
            window = image[y - window_size_y:y, x - window_size_x:x]
            windows.append(window)
    windows = np.array(windows)
    n_windows = windows.shape[0]

    # For all windows, calculate 2D Fourier Transformation
    windows_fft = np.abs(np.fft.fft2(windows))
    windows_fft_shifted = np.fft.fftshift(windows_fft**2) # This takes extremely long

    # Flatten and get PCA
    windows_fft_flat = np.reshape(windows_fft_shifted,
                                  (n_windows,
                                   window_size_y*window_size_x))

    pca = PCA().fit(windows_fft_flat)
    evr = pca.explained_variance_ratio_

    return evr









