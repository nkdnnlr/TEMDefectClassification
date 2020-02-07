import os
import numpy as np
import scipy as sp
import scipy.signal as signal
import cv2 as cv2
import matplotlib.image as img
import matplotlib.pyplot as plt


def fourier_spectrum(filepath):
    """

    :param filepath:
    :return:
    """
    img = cv2.imread(filepath, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.show()


def high_pass_filter(filepath, mask):
    """

    :param filepath:
    :param mask:
    :return:
    """

    img = cv2.imread(filepath, 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift_cropped = fshift.copy()
    fshift_cropped[crow - mask : crow + mask, ccol - mask : ccol + mask] = 0

    magnitude_spectrum_cropped = 20 * np.log(np.abs(fshift_cropped))

    f_ishift = np.fft.ifftshift(fshift_cropped)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(221), plt.imshow(img, cmap="gray")
    plt.title("Input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_back, cmap="gray")
    plt.title("Image after HPF"), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(magnitude_spectrum_cropped, cmap="gray")
    plt.title("Magnitude Spectrum cropped"), plt.xticks([]), plt.yticks([])
    plt.show()

    return img_back


def low_pass_filter(filepath, mask=20):
    """

    :param filepath:
    :param mask:
    :return:
    """

    img = cv2.imread(filepath, 0)
    f = np.fft.fft2(img)

    f_real = np.real(f)
    f_im = np.imag(f)

    print(f_im)
    # exit()

    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # ind = np.matrix.argpartition(magnitude_spectrum, -4)
    # # print(ind)
    # # exit()

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift_cropped = fshift.copy()
    fshift_cropped[0 : crow - mask, :] = 0
    fshift_cropped[crow + mask : rows, :] = 0
    fshift_cropped[:, 0 : ccol - mask] = 0
    fshift_cropped[:, ccol + mask : cols] = 0

    magnitude_spectrum_cropped = 20 * np.log(np.abs(fshift_cropped))

    f_ishift = np.fft.ifftshift(fshift_cropped)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # normalized = np.zeros(img_back.shape)
    # normalized = cv2.normalize(img_back, normalized)
    # img_back = normalized

    plt.subplot(221), plt.imshow(img, cmap="gray")
    plt.title("Input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(f_real)  # , cmap="gray", vmin=-1e6, vmax=1e6)
    # plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(f_im)  # , cmap="gray", vmin=-1e6, vmax=1e6)
    # plt.title("Image after LPF"), plt.xticks([]), plt.yticks([])

    # plt.subplot(222), plt.imshow(magnitude_spectrum, cmap="gray")
    # plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(img_back, cmap="gray")
    # plt.title("Image after LPF"), plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(magnitude_spectrum_cropped, cmap="gray")
    # plt.title("Magnitude Spectrum cropped"), plt.xticks([]), plt.yticks([])
    plt.show()

    return img_back


def linescan(img):
    print(img.shape)
    for i in range(0, 1024, 50):
        if i == 100 or i == 800:
            plt.plot(-1.5 * i + img[i, :], label=str(i), c="r")
        else:
            plt.plot(-1.5 * i + img[i, :], label=str(i), c="gray")
    # plt.legend()
    plt.yticks([])
    plt.xlabel("Pixel in x")
    plt.ylabel("Line scan intensity")
    plt.show()


def run():
    img_dir = "../../Data/Texture/KTH TIPS2b/KTH-TIPS2-b/cotton/sample_b"
    img_dir = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/data/"
    img_dir = "../data/training/0"

    img_dir = "../../data/cubic/defective/fold0/train/defective/images"

    assert os.path.isdir(img_dir)

    i = 0
    for subdir, dirs, files in os.walk(img_dir):
        print("i:", i)
        i += 1
        j = 0
        for file in files:
            print(file)
            print("j:", j)
            j += 1
            filepath = subdir + os.sep + file

            if j >= 8:
                exit()

            # if filepath.endswith(".png"):
            if filepath.endswith(".tif"):

                print(filepath)
                # high_pass_filter(filepath, mask=20)
                fourier_spectrum(filepath)
                # img = low_pass_filter(filepath, mask=30)

                # linescan(img)
                # exit()

    print("Done")


run()
