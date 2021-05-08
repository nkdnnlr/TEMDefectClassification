"""
Highly inspired by the work of Scotts Lab: https://arxiv.org/abs/2001.05022v1
https://github.com/ScottLabUCB/HTTEM/blob/master/pyNanoFind/bragg_filtering/bragg_filtering.ipynb
"""

import numpy as np
import scipy.fftpack as ftp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import medfilt2d
from skimage.util import pad
import sys
import h5py
# import realDataProcess as rdp
# import freqCutting as fcut
from scipy import fftpack
from scipy import signal
from skimage.morphology import opening, closing, square
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import helpers as helpers

def circular_mask(size_x=1024, size_y=1024, cx=512, cy=512, r=50):
    x = np.arange(0, size_x)
    y = np.arange(0, size_y)
    arr = np.zeros((size_x, size_y))
    mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
    arr[mask] = 1
    return arr
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(x, y, arr)
    # plt.colorbar()
    # plt.show()

def ring_mask(size_x=1024, size_y=1024, cx=512, cy=512, r_in=50, r_out=100):
    x = np.arange(0, size_x)
    y = np.arange(0, size_y)
    arr = np.zeros((size_x, size_y))
    mask_out = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r_out ** 2
    arr[mask_out] = 1
    mask_in = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r_in ** 2
    arr[mask_in] = 0
    return arr




# def imm(img_tensor, colorbar = True):
#     if len(img_tensor.shape) > 2:
#         if len(img_tensor.shape) == 4:
#             size = (img_tensor.shape[1],img.shape[2])
#         elif len(img_tensor.shape) == 3:
#             size = (img_tensor.shape[0],img.shape[1])
#     else:
#         size = img_tensor.shape
#     plt.figure(figsize=(10,10))
#     plt.imshow(img_tensor.reshape(size),cmap='viridis')
#     if colorbar == True:
#         plt.colorbar()
#     plt.show()

def immFFT(fftpic):
    plt.figure(figsize=(10,10))
    plt.imshow(np.real(np.sqrt(np.square(fftpack.fftshift(fftpic)))).astype('uint8'))
    plt.axis('off')
    plt.show()

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def bragg_filter_amorphous(img, ksize=9, sig=1, shape=(1024, 1024), cut_edge=(50, 50), r_mask=50):
    """"
    Segmentation of amorphous regions by Bragg-filtering, using a circular mask in Fourier space.
    """
    fft = ftp.fftshift(ftp.fft2(img.reshape(img.shape[0], img.shape[1])))
    intf = np.log(abs(fft) ** 2)
    intfilt = medfilt2d(intf, kernel_size=25)
    dif = intf - intfilt
    dif[dif < 2 * intfilt.std()] = 0
    kernel = gkern(ksize, sig)
    peaks = signal.convolve2d(dif, kernel)
    cut_pix_x = (peaks.shape[0] - shape[0]) // 2
    cut_pix_y = (peaks.shape[1] - shape[1]) // 2
    peaks = peaks[cut_pix_x:-cut_pix_x, cut_pix_y:-cut_pix_y]

    peaks[:cut_edge[0], :] = 0
    peaks[-cut_edge[0]:, :] = 0
    peaks[:, :cut_edge[1]] = 0
    peaks[:, -cut_edge[1]:] = 0

    peaks = peaks * circular_mask(r=r_mask)

    filtered_fft = fft * peaks

    img = abs(ftp.ifft2(filtered_fft)) ** 2
    img -= img.min()

    img[img < 20000000] = 0
    img[img > 20000000] = 1

    return img

def bragg_filter_symmetry(img, ksize=9, sig=1, shape=(1024, 1024), cut_edge=(50, 50), n_peaks=6, thr1=0.05, thr2=0.5):

    fft = ftp.fftshift(ftp.fft2(img.reshape(img.shape[0], img.shape[1])))
    intf = np.log(abs(fft) ** 2)
    intfilt = medfilt2d(intf, kernel_size=25)

    dif = intf - intfilt
    dif[dif < 2 * intfilt.std()] = 0
    kernel = gkern(ksize, sig)
    peaks = signal.convolve2d(dif, kernel)
    cut_pix_x = (peaks.shape[0] - shape[0]) // 2
    cut_pix_y = (peaks.shape[1] - shape[1]) // 2
    peaks = peaks[cut_pix_x:-cut_pix_x, cut_pix_y:-cut_pix_y]
    peaks[:cut_edge[0], :] = 0
    peaks[-cut_edge[0]:, :] = 0
    peaks[:, :cut_edge[1]] = 0
    peaks[:, -cut_edge[1]:] = 0

    # If this is in, then the amorphous region will be highlighted
    peaks_wo_center = peaks.copy()
    peaks_wo_center[peaks.shape[0]//2-10:peaks.shape[0]//2+10, peaks.shape[1]//2-10:peaks.shape[1]//2+10] = 0

    thr = np.max(peaks_wo_center)*0.5
    peaks[peaks < thr] = 0
    peaks[peaks > thr] = 1
    smoother_peaks = signal.convolve2d(peaks, kernel)
    smoother_peaks = smoother_peaks[cut_pix_x:-cut_pix_x, cut_pix_y:-cut_pix_y]
    inv_peaks = smoother_peaks.copy()
    inv_peaks += 1
    inv_peaks[inv_peaks > 1] = 0
    inv_peaks = signal.convolve2d(inv_peaks, kernel)
    # inv_peaks = inv_peaks[cut_pix_x:-cut_pix_x, cut_pix_y:-cut_pix_y]
    filtered_fft = fft * smoother_peaks
    # inv_filtered_fft = fft * inv_peaks

    isolated_peaks = isolate_bragg_peaks(filtered_fft, peak_thresh=50)

    final_seg = np.zeros((peaks.shape[0], peaks.shape[1]))
    for idx, img in enumerate(isolated_peaks):
        if idx == n_peaks:  # Get strongest peaks only
            break

        img = abs(ftp.ifft2(img)) ** 2
        img -= img.min()
        img = img / img.max()
        img[img < thr1] = 0
        final_seg += img
    final_seg = final_seg / final_seg.max()
    final_seg[final_seg > thr2] = 1
    final_seg[final_seg < 1] = 0

    return final_seg


def bragg_seg(filt_fft):
    """Segmentation of fourier transform"""
    ffft = ftp.fftshift(filt_fft)
    half_point = filt_fft.shape[0] // 2
    ffft[half_point:, :] = 0
    seg_map = ftp.ifft2(ffft)
    return ffft, seg_map

def clear_border(x, border):
    x[:, :border] = 0
    x[:, -border:] = 0
    x[:border, :] = 0
    x[-border:, :] = 0
    return x


def isolate_bragg_peaks(filt_fft, peak_thresh=50, plot=False):
    """Inputting a filtered fft from the bragg filter function returns array of segmentation maps based on each identified bragg peak"""
    testf, testr = bragg_seg(filt_fft)

    image = np.real(np.sqrt(ftp.fftshift(testf.copy()) ** 2)).astype('uint8')

    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    # cleared = bw
    cleared = clear_border(bw, 20)

    # label image regions
    label_image = label(cleared)  # Connected component labeling

    # if plot == True:
    # image_label_overlay = label2rgb(label_image, image=image)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

        # fig, ax = plt.subplots(2)
        # ax[0].imshow(image)
        # ax[1].imshow(label_image)

    bragg_spots = []
    region_areas = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        # draw rectangle around segmented coins
        if region.area > peak_thresh:
            minr, minc, maxr, maxc = region.bbox
            filt = ftp.fftshift(testf.copy())
            filt[:minr, :] = 0
            filt[maxr:, :] = 0
            filt[:, :minc] = 0
            filt[:, maxc:] = 0
            bragg_spots.append(filt)
            region_areas.append(region.area)

    idx = np.flip(np.argsort(region_areas))
    region_areas = np.array(region_areas)[idx]
    bragg_spots = np.array(bragg_spots)[idx]


    print(region_areas)
    return bragg_spots


def plot_ifft3(fft):
    img = abs(ftp.ifft2(fft)) ** 2
    img -= img.min()
    img = img / img.max()
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='viridis')
    return img
