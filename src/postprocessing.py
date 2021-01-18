import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import frangi, hessian

from utils.preprocessing import preprocess_image, normalize_intensity, cut_intensity, gaussian_filtering

output_dir = '/home/nik/UZH/IBM/TEMDefectClassification/output/eigenfilter_segmentation/b12_fs11'


# image_names = ['JEOL BF 05 SIH09 no annotation07_segmented.npy', 'JEOL BF 06_lou no annotation07_segmented.npy']

m = 100


for name in os.listdir(output_dir):
    if not name.endswith('.npy'):
        continue
    filepath = os.path.join(output_dir, name)
    image = np.load(filepath)    
    image_cut = cut_intensity(image, min=0, max=22)
    image_normalized = normalize_intensity(image_cut)
    image_filtered = gaussian_filtering(image_normalized*255, kernel_size=101, stdev=6)/255.
    image_smaller = image_filtered[m:-m, m:-m]
    # th, image_binarized = cv2.threshold(np.image_smaller, 128, 255, cv2.THRESH_OTSU)#frangi(image_smaller)
    # th, image_binarized = cv2.threshold(np.uint8(255*image_smaller),127,255,cv2.THRESH_OTSU)
    th, image_binarized = cv2.threshold(np.uint8(255*image_smaller),127,255,cv2.THRESH_TRIANGLE)


      
    
    # K-MEANS CLUSTERING
    # image_smaller_ = image_smaller.reshape(image_smaller.shape[0] * image_smaller.shape[1], 1)
    # from sklearn.cluster import KMeans, MiniBatchKMeans, Birch

    # n=2
    # kmeans = MiniBatchKMeans(n_clusters=n).fit(image_smaller_)
    # clustered = kmeans.cluster_centers_[kmeans.labels_]
    # labels = kmeans.labels_
    # # for n in range(n):
    # n=0
    # image_cluster = []
    # for i in range(len(labels)):
    #     if(labels[i]) == n:
    #         image_cluster.append(float(clustered[i]))
    #     else:
    #         image_cluster.append(1)
    # if(n==1):
    #     image_fix= np.array(image_cluster).reshape(image_smaller.shape)
    # image_binarized = np.array(image_cluster).reshape(image_smaller.shape)
        # plt.imshow(reshape_clustered, cmap=plt.get_cmap("gray"),vmin=0, vmax=1)
        # plt.show()
    
    
    # Morphology
    #    defining the kernel i.e. Structuring element 
    # kernel = np.ones((12, 12), np.uint8) 
    # #     # defining the closing function  
    # image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_OPEN, kernel) 
    # image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_CLOSE, 2*kernel) 
    
    # image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_OPEN, 50*kernel) 
    # image_binarized = cv2.morphologyEx(image_binarized, cv2.MORPH_CLOSE, 40*kernel) 
    image_binarized = cv2.medianBlur(image_binarized, 31) 
    
    fig, axes = plt.subplots(ncols = 2)
    axes[0].imshow(image)
    # axes[1].imshow(image_cut)
    # axes[1].imshow(image_normalized)
    # axes[1].imshow(image_smaller)
    # axes[2].imshow(image_binarized, cmap='gray')
    axes[1].imshow(image_binarized, cmap='gray')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(name)
    
    plt.savefig(os.path.join(output_dir, name[:-4]+'_binarized.png'))
    plt.show()
    plt.close()
    # exit()