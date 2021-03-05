import os
from pathlib import Path
import cv2
import json
import tqdm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.path import Path as mPath
import matplotlib.patches as mpatches

import src.utils.helpers as helpers


def get_mask(filename_json, int_dict = {'B': 0, 'S': 1, 'A': 2}, size_x=1024, size_y=1024):
    """
    Get pixel-wise numpy array with labels inside
    :param filename_json:
    :param int_dict:
    :param size_x:
    :param size_y:
    :return:
    """
    assert filename_json.endswith(".json")
    with open(filename_json, 'r') as file:
        json_data = json.load(file)
    mask_total = np.zeros((size_y, size_x))
    for item in json_data["shapes"]:
        label = item['label']
        points_polygon = np.array(item["points"])

        x, y = np.meshgrid(np.arange(size_y), np.arange(size_x))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        p = mPath(points_polygon)  # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(size_y, size_x) * int_dict[label]  # now you have a mask with points inside a polygon
        mask_total = np.maximum.reduce([mask, mask_total])
    return mask_total

# def get_borders(mask):


def get_masks(dir_json,
                dir_labels,
                int_dict = {'B': 0, 'S': 1, 'A': 2}):
    """
    Get masks and save as tif file
    :param dir_json:
    :param dir_labels:
    :param int_dict:
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
        file for file in os.listdir(dir_json) if file.endswith(".json")
    ]

    for i in tqdm.trange(len(filenames)):
        filename = filenames[i]
        full_path = os.path.join(dir_json, filename)
        full_path_label = os.path.join(dir_labels, filename[:-5]+".tif")
        label = get_mask(full_path, int_dict)
        # label = np.uint8(np.round((get_mask(full_path, int_dict)*255.0)))
        #
        # print(label.shape)
        # print(np.unique(label))
        # print(full_path_label)
        # Image.fromarray(np.uint8(label * 255.0)).save(full_path_label)
        cv2.imwrite(full_path_label, np.uint8(label * 255.0))
        # image = cv2.imread(full_path_label)
        # print(np.unique(image))
        # label.astype('int8').tofile(full_path_label)
        # image = Image.fromarray(label)
        # print(image.shape)
        # print(np.unique(image))
        # exit()
        # Image.fromarray(np.uint8(label * 255.0)).save(full_path_label)


def plot_labels(source_dir,
                output_dir,
                int_dict = {'B': 0, 'S': 1, 'A': 2},
                label_dict = {'B': 'primary symmetry', 'S': 'secondary symmetry', 'A': 'blurred'},
                show=False):
    """
    Make plots comparing raw data with labeled data
    :param source_dir:
    :param output_dir:
    :param int_dict:
    :param label_dict:
    :param show:
    :return:
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith(".json"):
            name = filename[:-5]
            filename_json = filename
            filename_tif = filename[:-5]+".tif"
            print(filename_tif + " :" )

            fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(12,6))
            original = helpers.get_image(os.path.join(source_dir, filename_tif))
            labelmask = get_mask(filename_json=os.path.join(source_dir, filename_json), int_dict=int_dict)

            axes[0].imshow(original, cmap="gray")
            axes[1].imshow(original, cmap="gray")
            im = axes[1].imshow(labelmask, alpha=0.2, cmap='hot', vmax=max(int_dict.values())*1.2)

            axes[0].set_title("raw")
            axes[1].set_title("labeled")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            values = np.unique(labelmask.ravel())
            # get the colors of the values, according to the
            # colormap used by imshow
            int_dict_reversed = {v: k for k, v in int_dict.items()}
            patches = [mpatches.Patch(color=im.cmap((value/max(int_dict.values()))/1.2), label=label_dict[int_dict_reversed[value]]) for
                       value in
                       values]
            # put those patched as legend-handles into the legend
            axes[1].legend(handles=patches, bbox_to_anchor=(0.73, -0.02), loc=0, borderaxespad=0.)

            plt.suptitle(name)
            plt.savefig(output_dir.joinpath(name + "_labeled.png"), dpi=300)
            # plt.show()
            plt.close()



if __name__ == '__main__':
    source_dir = Path("../../data/all_defective")  # os.path.join(os.getcwd(), )
    output_dir = Path("../../data/all_defective/labeled")  # os.path.join(os.getcwd(), )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_labels(source_dir=source_dir, output_dir=output_dir)
