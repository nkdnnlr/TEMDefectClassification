import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import frangi, hessian

from utils.preprocessing import preprocess_image, normalize_intensity, cut_intensity, gaussian_filtering
from utils import helpers

# output_dir = '/home/nik/UZH/IBM/TEMDefectClassification/output/eigenfilter_segmentation/b12_fs11'
# output_dir = '/home/nik/UZH/IBM/TEMDefectClassification/output/eigenfilter_segmentation_forpost/b16_fs7'
output_dir = '/home/nik/UZH/IBM/TEMDefectClassification/output/eigenfilter_segmentation_12/b16_fs7'


# image_names = ['JEOL BF 05 SIH09 no annotation07_segmented.npy', 'JEOL BF 06_lou no annotation07_segmented.npy']


for name in os.listdir(output_dir):
    if not name.endswith('symmetry.npy'):
        continue
    print(name)
    filepath_sym = os.path.join(output_dir, name)
    filepath_var = filepath_sym[:-12]+"_08_localvariance.npy"
    filepath_bestpatch = filepath_sym[:-12]+"bestpatch.npy"

    image_var = np.load(filepath_var)
    image_sym = np.load(filepath_sym)
    best_patch = np.load(filepath_bestpatch)

    # Local Variance Filtering
    localvariance_patch = helpers.localvariance_filter(image=best_patch)
    minlocalvariance = np.min(localvariance_patch)
    image_var_filtered = gaussian_filtering(image_var*255, kernel_size=101, stdev=10)/255.
    image_var_binarized = image_var_filtered<minlocalvariance/1.2

    # Symmetry Filtering
    minval = np.min(image_sym[image_sym>0])
    image_cut = cut_intensity(image_sym, min=minval, max=None)
    image_normalized = normalize_intensity(image_cut)
    image_filtered = gaussian_filtering(image_normalized*255, kernel_size=101, stdev=6)/255.
    image_smaller = image_filtered
    th, image_binarized = cv2.threshold(np.uint8(255*image_smaller),127,255,cv2.THRESH_TRIANGLE)
    image_binarized = cv2.medianBlur(image_binarized, 21)
    image_binarized = image_binarized / np.max(image_binarized)
    binarized_both = np.maximum.reduce([image_var_binarized*1.05, image_binarized])

    # Plots
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(image_var)
    axes[1].imshow(image_sym)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(name)
    plt.show()

    fig, axes = plt.subplots(ncols=1)
    axes.imshow(binarized_both, cmap='gist_stern_r', vmin=0, vmax=1.05)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.suptitle(name)
    plt.savefig(os.path.join(output_dir, name[:-4]+'_binarized.png'))
    plt.show()

    continue

    exit()
    # exit()