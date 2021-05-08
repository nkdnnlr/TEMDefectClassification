import os
from PIL import Image
import tqdm

import numpy as np
from numpy import percentile
import cv2
from skimage import exposure, filters
import matplotlib.pyplot as plt

# from src.utils.helpers import get_image
from src.utils.helpers import get_image

def mean_filtering(image, kernel_size=5):
    """
    Convolves averaging filter of a given kernel size with image.
    :param image: np.array of OpenCV format (in [0, 255])
    :param kernel_size: size of convoluting mask
    :return: np.array of OpenCV format
    """
    return cv2.blur(image, ksize=(kernel_size, kernel_size))


def median_filtering(image, kernel_size=5):
    """
    Convolves median filter of a given kernel size with image.
    :param image: np.array of OpenCV format (in [0, 255])
    :param kernel_size: size of convoluting mask
    :return: np.array of OpenCV format
    """
    return cv2.medianBlur(image, ksize=kernel_size)


def gaussian_filtering(image, kernel_size=5, stdev=1):
    """
    Convolves Gaussian filter of a given kernel size and standard deviation with image
    :param image: np.array of OpenCV format (in [0, 255])
    :param kernel_size: size of convoluting mask
    :param stdev: standard deviation for Gaussian
    :return: np.array of OpenCV format
    """
    return cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=stdev)


def laplacian_filtering(image, kernel_size=19):
    """
    Convolves Laplacian filter of a given kernel size with image and subsequently flips image intensities.
    Here we choose a rather large kernel size, which often results in reducing the low-frequency intensity changes.
    :param image: np.array of OpenCV format (in [0, 255])
    :param kernel_size: size of convoluting mask. Choose large value for high-pass-filtering
    :return: np.array of OpenCV format
    """
    return 255-cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)


def rescale_intensity(image, p_range=(2, 98)):
    """
    Return image after stretching or shrinking its intensity levels to desired percentile.
    Often being done for single patch.
    :param image: np.array
    :param p_range: Range in percents
    :return: np.array
    """
    p2, p98 = percentile(image, p_range)
    return exposure.rescale_intensity(image, in_range=(p2, p98))

def cut_intensity(image, min=None, max=None):
    """

    Args:
        image (np.array): 
        min ([type]): [description]
        max ([type]): [description]
    """
    image_cut = image.copy()
    if min is not None:
        image_cut[image_cut < min] = min
    if max is not None:
        image_cut[image_cut > max] = max
    return image_cut    
    
    
def normalize_intensity(image, max=None):
    """Normalize the intensity to range [0, 1]
    Args:
        image ([type]): Numpy array of image
    """
    image = (image - np.min(image))
    if max is None:
        image = image / np.max(image)
    else:
        if max>np.max(image):
            print("now")
            image = image / max
        else:
            image = image / np.max(image)
    return image

def preprocess_image(image,
                     lowpass_filter="gaussian", lowpass_kernel_size=5,
                     highpass_filter="laplacian", highpass_kernel_size=19,
                     rescale=False):
    """
    Preprocesses image with high highpass, lowpass and/or rescaling intensity
    :param image:
    :param lowpass_filter: type=str; either "mean", "median", "gaussian" or None
    :param lowpass_kernel_size: size of convoluting mask
    :param highpass_filter: type=str; either "laplacian" or None
    :param highpass_kernel_size: size of convoluting mask
    :param rescale: type=bool; rescale intensity or not.
    :return:
    """
    lowpass_filters = {"mean": mean_filtering,
                       "median": median_filtering,
                       "gaussian": gaussian_filtering,
                       }

    highpass_filters = {"laplacian": laplacian_filtering,
                        }

    image_preprocessed = image

    try:
        if highpass_filter is not None:
            image_preprocessed = highpass_filters[highpass_filter](image_preprocessed, highpass_kernel_size)
        if lowpass_filter is not None:
            image_preprocessed = lowpass_filters[lowpass_filter](image_preprocessed, lowpass_kernel_size)
    except KeyError:
        print("Filter not implemented yet.")

    if rescale:
        image_preprocessed = rescale_intensity(image_preprocessed)

    image_preprocessed = np.interp(image_preprocessed, (image_preprocessed.min(), image_preprocessed.max()), (0, 254))

    return image_preprocessed


def preprocess_file(file_path, file_path_preprocessed,
                    lowpass_filter="gaussian", lowpass_kernel_size=5,
                    highpass_filter="laplacian", highpass_kernel_size=19,
                    rescale=False, plot=False):
    """
    Preprocess an image at a given location
    :param file_path:
    :param file_path_preprocessed:
    :param lowpass_filter:
    :param lowpass_kernel_size:
    :param highpass_filter:
    :param highpass_kernel_size:
    :param rescale:
    :return:
    """
    image = get_image(file_path)
    # image_preprocessed = preprocess_image(image=image, lowpass_filter=lowpass_filter, lowpass_kernel_size=lowpass_kernel_size, highpass_filter=highpass_filter, highpass_kernel_size=highpass_kernel_size, rescale=rescale)

    image_preprocessed = filters.gaussian(image, 2) - filters.gaussian(image, 3)  # Looks quite much like
    # previous setting
    image_preprocessed = np.interp(image_preprocessed, (image_preprocessed.min(), image_preprocessed.max()), (0, 254))

    image_preprocessed =  image_preprocessed.astype('uint8')
    cv2.imwrite(file_path_preprocessed, image_preprocessed)



def preprocess_folder(directory_original, directory_preprocessed, image_format=".tif",
                      lowpass_filter="gaussian", lowpass_kernel_size=5,
                      highpass_filter="laplacian", highpass_kernel_size=19,
                      rescale=False):
    """
    Preprocesses all images with a given format in a directory, according to choices of preprocessing.
    :param directory_original: directory with images to be preprocessed
    :param directory_preprocessed: target directory
    :param image_format: type of image. so far only tested with ".tif"
    :param lowpass_filter: type=str; either "mean", "median", "gaussian" or None
    :param lowpass_kernel_size: size of convoluting mask
    :param highpass_filter: type=str; either "laplacian" or None
    :param highpass_kernel_size: size of convoluting mask
    :param rescale: type=bool; rescale intensity or not.
    :return:
    """
    print(directory_original)
    assert os.path.isdir(directory_original)
    paths = os.listdir(directory_original)
    paths.sort()

    print("preprocessed: ", directory_preprocessed)
    if not os.path.isdir(directory_preprocessed):
        os.mkdir(directory_preprocessed)

    for i in tqdm.trange(len(paths)):
        path = paths[i]
        print(path)
        if not path.endswith(image_format):
            continue

        file_path = os.path.join(directory_original, path)
        file_path_preprocessed = os.path.join(directory_preprocessed, path)

        preprocess_file(file_path, file_path_preprocessed,
                        lowpass_filter=lowpass_filter, lowpass_kernel_size=lowpass_kernel_size,
                        highpass_filter=highpass_filter, highpass_kernel_size=highpass_kernel_size,
                        rescale=rescale)
        # TODO: Make sure that arguments are passed on, best in a nice way!!!




def preprocess_showingoff(file_path, output_dir, name,
                    lowpass_filter="gaussian", lowpass_kernel_size=5,
                    highpass_filter="laplacian", highpass_kernel_size=19,
                    rescale=False, plot=False):
    """
    Preprocess an image at a given location and plot results
    :param file_path:
    :param file_path_preprocessed:
    :param lowpass_filter:
    :param lowpass_kernel_size:
    :param highpass_filter:
    :param highpass_kernel_size:
    :param rescale:
    :return:
    """
    image = get_image(file_path)
    image_laplace = preprocess_image(image=image, lowpass_filter=None, lowpass_kernel_size=None,
                                     highpass_filter=highpass_filter, highpass_kernel_size=highpass_kernel_size)
    image_preprocessed = preprocess_image(image=image_laplace, lowpass_filter=lowpass_filter,
                                          lowpass_kernel_size=lowpass_kernel_size, highpass_filter=None,
                                          highpass_kernel_size=None)


    plt.figure(figsize=(20, 20))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    plt.imshow(image_laplace, cmap='gray')
    plt.title("Laplacian")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    plt.imshow(image_preprocessed, cmap='gray')
    plt.title("Laplacian + Gaussian")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name + "preprocessing.png"), dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # directory_original = "/home/nik/Documents/Defect Classification/Data/cubic/defective/images"
    # directory_preprocessed = "/home/nik/Documents/Defect Classification/Data/cubic/defective/preprocessed"
    # preprocess_folder(directory_original=directory_original, directory_preprocessed=directory_preprocessed)

    # file1 = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/data/cubic/cnn/train/defective/7.tif"
    # preprocess_showingoff(file1)
    # output_dir = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/output/preprocessing"
    # file = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/data/cubic/defective/images/JEOL BF 50_SIH05 no annotation.tif"
    # file2 = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/data/cubic/defective/folds/fold0/train/original/images/0/JEOL BF 05 SIH09 no annotation.tif"
    # preprocess_showingoff(file, output_dir=output_dir, name="JEOL BF 50_SIH05")


    image = np.array([2,7,16,4,24])
    print(image)
    
    image = cut_intensity(image, 6, 20)
    print(image)
    image = normalize_intensity(image)
    print(image)
    
    exit()
