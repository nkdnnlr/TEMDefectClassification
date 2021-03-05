import os
import sys
from PIL import Image

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import tqdm

def get_labels_from_json(dir_labels, dir_annotated, format="tif", no_label=False):
    if not os.path.exists(dir_labels):
        print("New Label Directory:", dir_labels)
        os.makedirs(dir_labels)
    else:
        print("Existing Label Directory:", dir_labels)
        print("Attention: Overwriting...")

    print("Create labels in directory ", dir_labels)

    filenames = [
        file for file in os.listdir(dir_annotated) if file.endswith("." + format)
    ]

    for i in tqdm.trange(len(filenames)):
        filename = filenames[i]
    # for filename in filenames:
        full_path = os.path.join(dir_annotated, filename)
        full_path_label = os.path.join(dir_labels, filename)
        image = img.imread(full_path)

        label = extract_label(
            image, labelcolors=["red", "green", "blue"], fill=True, no_label=no_label
        )

        Image.fromarray(np.uint8(label * 255.0)).save(full_path_label)

        import matplotlib.gridspec as gridspec
        plt.figure(figsize=(16, 8))
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.05, hspace=0.05)

        ax0 = plt.subplot(gs1[0])
        ax0.imshow(image, cmap='gray')
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = plt.subplot(gs1[1])
        ax1.imshow(label, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])

        plt.show()


def extract_label(
    image,
    labelcolors=["red", "green", "blue"],
    fill=True,
    noise_suppression=True,
    no_label=False,
):
    """
    For an annotated image, extracts binary label containing the label contour.
    :param image_path: Path to grayscale image with drawn labels.
    :param labelcolors: Color of the label. So far only red is possible
    :param fill: boolean. If true, fills out contours.
    :param noise_suppression: If true, filters binary mask with Gaussian kernel to reduce noise.
    :param no_label: If true, creates labels that are all zeros (black). Good&fast for non-defective images.
    :return: np array with shape [height, width]
    """
    height, width, channels = image.shape

    label = np.zeros([height, width], dtype=np.uint8)
    if no_label:
        return label

    for color in labelcolors:
        if color == "red":
            rgb_threshold = [220, -200, -200]
        elif color == "green":
            rgb_threshold = [-200, 220, -200]
        elif color == "blue":
            rgb_threshold = [-200, -200, 220]
        else:

            print("Label color should be 'red', 'green' or 'blue'.")
            return

        s = []
        for color_th in rgb_threshold:
            s.append(color_th / abs(color_th))

        for py in range(height):
            for px in range(width):
                if (
                    s[0] * image[py, px, 0] > rgb_threshold[0]
                    and s[1] * image[py, px, 1] > rgb_threshold[1]
                    and s[2] * image[py, px, 2] > rgb_threshold[2]
                ):
                    label[py, px] = 1

    if not fill:
        if noise_suppression:
            label = cv2.medianBlur(label, 3)
            pass
        return label

    else:
        label_filled = fill_holes(label.copy())

        if noise_suppression:
            label_filled = cv2.medianBlur(label_filled, 3)
            pass
        label_filled = label_filled * 1.0 / 255.0

        return label_filled


def fill_holes(im_th):
    """
    Fill holes in contour using the floodfill algorithm.
    :param im_th: Binary thresholded image.
    :return: Output image.
    """
    # Copy the thresholded image.
    im_floodfill = im_th.copy() * 254

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels larger than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def get_labels(dir_labels, dir_annotated, format="tif", no_label=False):
    """
    Creates binary pixel-wise labels from annotations.
    :param dir_labels: Target directory where labels will be saved. Will be created if not existing.
    :param dir_annotated: Directory where annotations are located.
        ATTENTION: Annotations must ...
            * ... be of the color RED ([255, 0, 0]), GREEN ([0, 255, 0]) or BLUE ([0, 0, 255])
            * ... be continuous lines (no dashes!) that form a closed loop
            * ... not touch the walls or be so close to the walls that there is no space from them to flow.
        Let me explain this further. You can imagine the algorithm to work like water flowing from the walls,
        only halting of borders of a specific color.
        Everything covered by the water will be black (zeros), everything else white (ones).
        If it didn't work for your annotation, try again for them fulfilling above points.
    :param format: Image format, tried with "tif"
    :param no_label: If true, creates labels that are all zeros (black). Good&fast for non-defective images.
    :return:
    """
    if not os.path.exists(dir_labels):
        print("New Label Directory:", dir_labels)
        os.makedirs(dir_labels)
    else:
        print("Existing Label Directory:", dir_labels)
        print("Attention: Overwriting...")

    print("Create labels in directory ", dir_labels)

    filenames = [
        file for file in os.listdir(dir_annotated) if file.endswith("." + format)
    ]

    for i in tqdm.trange(len(filenames)):
        filename = filenames[i]
    # for filename in filenames:
        full_path = os.path.join(dir_annotated, filename)
        full_path_label = os.path.join(dir_labels, filename)
        image = img.imread(full_path)

        label = extract_label(
            image, labelcolors=["red", "green", "blue"], fill=True, no_label=no_label
        )

        Image.fromarray(np.uint8(label * 255.0)).save(full_path_label)

        import matplotlib.gridspec as gridspec
        plt.figure(figsize=(16, 8))
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.05, hspace=0.05)

        ax0 = plt.subplot(gs1[0])
        ax0.imshow(image, cmap='gray')
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = plt.subplot(gs1[1])
        ax1.imshow(label, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])

        plt.show()


if __name__ == "__main__":
    # Get labels for defective images (will take a while)
    get_labels(
        dir_labels="../../data/cubic/defective/labels/",
        dir_annotated="../../data/cubic/defective/annotations",
        no_label=False,
    )

    # Get labels for non-defective images (should be fast)
    get_labels(
        dir_labels="../../data/cubic/non_defective/labels/",
        dir_annotated="../../data/cubic/non_defective/annotations",
        no_label=True,
    )
