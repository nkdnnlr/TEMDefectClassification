import os
import sys
import argparse
import re
from zipfile import ZipFile
from pathlib import Path

from src.utils.get_labelsmask import get_labels
from src.utils.json2mask import get_masks, plot_labels
from src.utils.run_preprocessing import run
from src.train_cnn import train_cnn

parser = argparse.ArgumentParser()
# Data structure arguments
parser.add_argument("-file", "--file", type=str, default="all_data.zip",
                    help="Name of zipped data file.")
parser.add_argument("-unzip", "--unzip", type=bool, default=False,
                    help="Unzip?")
parser.add_argument("-labels", "--labels", type=bool, default=False,
                    help="Convert annotations to labels?")
parser.add_argument("-preprocessing", "--preprocessing", type=bool, default=True,
                    help="Preprocessing?.")
args = parser.parse_args()

UNZIP = args.unzip
LABELS = args.labels
PREPROCESSING = args.preprocessing
TRAIN_CNN = True
DATA_ZIPFILE = f'data/{args.file}'

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

# # Overwrite DATA_ZIPFILE with parsing argument
# if len(sys.argv) == 2:
#     DATA_ZIPFILE = sys.argv[-1]

# Specify Directories where data is or is going to be
parent_dir, name, image_format = re.split('\.|\/', DATA_ZIPFILE)
# print(parent_dir)

dir_data = os.path.join(parent_dir, name)

dir_defective = os.path.join(dir_data, 'defective')
dir_defective_images = os.path.join(dir_defective, 'images')
dir_defective_annotations = os.path.join(dir_defective, 'annotations')
dir_defective_labels = os.path.join(dir_defective, 'labels')
dir_defective_json = os.path.join(dir_defective, 'json')
# dir_defective_labels_blur = os.path.join(dir_defective, 'labels_blur')
# dir_defective_labels_symmetry = os.path.join(dir_defective, 'labels_symmetry')


dir_nondefective = os.path.join(dir_data, 'non_defective')
dir_nondefective_images = os.path.join(dir_nondefective, 'images')
dir_nondefective_annotations = os.path.join(dir_nondefective, 'annotations')
dir_nondefective_labels = os.path.join(dir_nondefective, 'labels')
# dir_nondefective_labels_blur = os.path.join(dir_nondefective, 'labels_blur')
# dir_nondefective_labels_symmetry = os.path.join(dir_nondefective, 'labels_symmetry')


if UNZIP:
    if image_format == 'zip':
        # Unzip Data
        with ZipFile(DATA_ZIPFILE, 'r') as zip_ref:
            zip_ref.extractall(parent_dir)

# if LABELS:
#     # Get labels for non-defective images (should be fast)
#     get_labels(
#         dir_labels=dir_nondefective_labels,
#         dir_annotated=dir_nondefective_annotations,
#         no_label=True,
#     )
#     # Get labels for defective images (will take a while)
#     get_labels(
#         dir_labels=dir_defective_labels,
#         dir_annotated=dir_defective_annotations,
#         no_label=False,
#     )

if LABELS:
    # Get labels for non-defective images (should be fast)
    # get_labels(
    #     dir_labels=dir_nondefective_labels_blur,
    #     dir_annotated=dir_nondefective_images,
    #     no_label=True,
    # )
    # get_labels(
    #     dir_labels=dir_nondefective_labels_symmetry,
    #     dir_annotated=dir_nondefective_images,
    #     no_label=True,
    # )
    get_labels(
        dir_labels=dir_nondefective_labels,
        dir_annotated=dir_nondefective_images,
        no_label=True,
    )
    # Get labels for defective images
    # For blurred labels...
    # get_masks(
    #     dir_labels=dir_defective_labels_blur,
    #     dir_json=dir_defective_json,
    #     int_dict={'B': 0, 'S': 0, 'A': 1},
    # )
    # # For symmery breaking labels...
    # get_masks(
    #     dir_labels=dir_defective_labels_symmetry,
    #     dir_json=dir_defective_json,
    #     int_dict={'B': 0, 'S': 1, 'A': 0},
    # )

    get_masks(
        dir_labels=dir_defective_labels,
        dir_json=dir_defective_json,
        int_dict={'B': 0, 'S': 0.5, 'A': 1},
    )


# plot_labels(source_dir=Path(dir_data).joinpath("defective_json"),
#             output_dir=Path(dir_data).joinpath("defective_labeled"),
#             int_dict = {'B': 0, 'S': 1, 'A': 2},
#             label_dict = {'B': 'primary symmetry', 'S': 'secondary symmetry', 'A': 'blurred'},
#             show=False)
# exit()

for n_train in [16, 14]:#range(2,10):
# while True:
#     n_train = 2
    dir_folds = os.path.join(dir_data, f'n_train_{n_train}')
    # dir_folds = os.path.join(f'output', name, 'test2', f'n_train_{n_train}')

    dir_output = os.path.join(f'output', name, f'n_train_{n_train}')

    print(dir_folds)
    print(dir_output)

    params = {
        "N_TRAIN": n_train,
        "N_TEST": 4,
        "N_FOLDS": 8,
        "TARGET_SIZE": 128,
        "AUGMENTATION_FACTOR": 40,  # Multiplying n_images by this factor with rotating and flipping
        "BATCH_SIZE": 100,  # Number of sampled patches
        "THRESHOLD_DEFECTIVE": 0.1,  # Lower limit for normalized defective area in order to be classified 'defective'
        "THRESHOLD_NONDEFECTIVE": 0.01,
        # Upper limit for normalized defective area in order to be classified 'non_defective'
    }

    if PREPROCESSING:
        run(dir_data, dir_defective, dir_nondefective, dir_folds, params)

    if TRAIN_CNN:
        train_cnn(dir_folds=dir_folds, output_dir=dir_output, n_folds=params["N_FOLDS"])



