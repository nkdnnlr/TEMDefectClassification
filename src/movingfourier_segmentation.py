import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from src.utils import helpers
from src.utils.preprocessing import preprocess_image


def get_variance_ratio(image):
    """
    Moves window over images and collects Fourier transformations.
    Then calculates PCA and returns list of explained variance for all PCs
    :param image: original image
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


def test_patches(patches):
    """
    Testing the patches with some metric
    :param patches:
    :return:
    """
    variance_ratios = []
    for patch in patches:
        variance_ratios.append(get_variance_ratio(image=patch))
    return np.array(variance_ratios)


def plot(image, result, vmax=None, output_path=None, cmap='inferno'):
    """
    Plot result nicely
    :param image:
    :param result:
    :param vmax:
    :param output_path:
    :param cmap:
    :return:
    """
    plt.figure(figsize=(20,20))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title("Original")

    plt.subplot(122)
    plt.imshow(result, vmin=0, vmax=vmax, cmap=cmap)
    plt.title("Defects")
    plt.colorbar()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    plt.show()


def test_file(file, target_size, step_size, output_path=None, index=1, name=None):
    """
    Testing file  with Moving Fourier method
    :param index:
    :param path:
    :param target_size:
    :param step_size:
    :return:
    """
    if name is None:
        name = index

    starttime = time.time()
    print("Getting file at path: {}".format(file))
    original = helpers.get_image(file)
    image = preprocess_image(original,
                             lowpass_filter="gaussian", lowpass_kernel_size=5,
                             highpass_filter="laplacian", highpass_kernel_size=19,
                             rescale=False)

    print("Finding patches...")

    target_size = target_size
    step_size = step_size
    patches = helpers.get_patches(image, target_size=target_size, step=step_size)
    print("Testing {} patches...".format(len(patches)))
    list_of_evr = test_patches(patches)

    scores = list_of_evr[:, 1] / list_of_evr[:, 0]# + list_of_evr[:, 2] / list_of_evr[:, 0]
    result1 = helpers.map_scores(image, scores, target_size=target_size, step_size=step_size)

    scores = list_of_evr[:, 2] / list_of_evr[:, 0]# + list_of_evr[:, 2] / list_of_evr[:, 0]
    result2 = helpers.map_scores(image, scores, target_size=target_size, step_size=step_size)

    scores = list_of_evr[:, 3] / list_of_evr[:, 0]# + list_of_evr[:, 2] / list_of_evr[:, 0]
    result3 = helpers.map_scores(image, scores, target_size=target_size, step_size=step_size)

    scores = list_of_evr[:, 2] / list_of_evr[:, 1]# + list_of_evr[:, 2] / list_of_evr[:, 0]
    result4 = helpers.map_scores(image, scores, target_size=target_size, step_size=step_size)

    scores = list_of_evr[:, 3] / list_of_evr[:, 1]# + list_of_evr[:, 2] / list_of_evr[:, 0]
    result5 = helpers.map_scores(image, scores, target_size=target_size, step_size=step_size)

    scores = list_of_evr[:, 3] / list_of_evr[:, 2]  # + list_of_evr[:, 2] / list_of_evr[:, 0]
    result6 = helpers.map_scores(image, scores, target_size=target_size, step_size=step_size)


    fig = plt.figure(figsize=(14, 6))
    gs1 = GridSpec(2, 5, left=0.01, right=0.99, wspace=0.02)

    ax0 = fig.add_subplot(gs1[:, 0:2], xticks=[0, 512, 1023], yticks=[0, 512, 1023])
    ax0.set_xlabel('x [px]')
    ax0.set_ylabel('y [px]')

    ax1 = fig.add_subplot(gs1[0, 2], xticks=[], yticks=[])
    ax2 = fig.add_subplot(gs1[0, 3], xticks=[], yticks=[])
    ax3 = fig.add_subplot(gs1[0, 4], xticks=[], yticks=[])
    ax4 = fig.add_subplot(gs1[1, 2], xticks=[], yticks=[])
    ax5 = fig.add_subplot(gs1[1, 3], xticks=[], yticks=[])
    ax6 = fig.add_subplot(gs1[1, 4], xticks=[], yticks=[])

    c = ax0.imshow(original, cmap='gray')
    ax0.text(0, 1.02, 'A', transform=ax0.transAxes,
            size=25, weight='bold')
    ax0.title.set_text(name)

    c = ax1.imshow(result1, cmap='viridis', vmin=0, vmax=10)
    fig.colorbar(c, ax=ax1)
    ax1.text(0, 1.03, 'B', transform=ax1.transAxes,
            size=25, weight='bold')

    c = ax2.imshow(result2, cmap='viridis', vmin=0, vmax=8)
    fig.colorbar(c, ax=ax2)
    ax2.text(0, 1.03, 'C', transform=ax2.transAxes,
            size=25, weight='bold')

    c = ax3.imshow(result3, cmap='viridis', vmin=0, vmax=2)
    fig.colorbar(c, ax=ax3)
    ax3.text(0, 1.03, 'D', transform=ax3.transAxes,
            size=25, weight='bold')

    c = ax4.imshow(result4, cmap='viridis', vmin=0, vmax=13)
    fig.colorbar(c, ax=ax4)
    ax4.text(0, 1.03, 'E', transform=ax4.transAxes,
            size=25, weight='bold')

    c = ax5.imshow(result5, cmap='viridis', vmin=0, vmax=2)
    fig.colorbar(c, ax=ax5)
    ax5.text(0, 1.03, 'F', transform=ax5.transAxes,
            size=25, weight='bold')

    c = ax6.imshow(result6, cmap='viridis', vmin=0, vmax=8)
    fig.colorbar(c, ax=ax6)
    ax6.text(0, 1.03, 'G', transform=ax6.transAxes,
            size=25, weight='bold')

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, "{}_fourierwindow.png".format(name)))
    plt.show()

    print("Maximas:")
    for res in [result1, result2, result3, result4, result5, result6]:
        print(res.max())
    print("Done in {}s.".format(time.time() - starttime))
    return


if __name__ == '__main__':
    TARGET_SIZE = 128
    # TARGET_SIZE = 224

    data_directory = "../data/cubic/defective/images/"
    assert os.path.isdir(data_directory)
    paths = os.listdir(data_directory)
    paths.sort()
    print(paths)

    output_dir = "../output/movingfourier_segmentation/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    TARGET_SIZE = 64
    STEP_SIZE = TARGET_SIZE // 4

    i = 0
    for path in paths:
        if not path.endswith(".tif"):
            continue
        name = path[:-4]
        file_path = os.path.join(data_directory, path)
        output_path = os.path.join(output_dir, path)
        test_file(file_path, target_size=TARGET_SIZE, step_size=STEP_SIZE, output_path=output_dir, index=i, name=name)
        i += 1
