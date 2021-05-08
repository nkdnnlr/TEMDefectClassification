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
parser.add_argument("-unzip", "--unzip", type=bool, default=True,
                    help="Unzip?")
parser.add_argument("-labels", "--labels", type=bool, default=True,
                    help="Convert annotations to labels?")
parser.add_argument("-preprocessing", "--preprocessing", type=bool, default=False,
                    help="Preprocessing?.")
parser.add_argument("-train_cnn", "--train_cnn", type=bool, default=False)
args = parser.parse_args()

UNZIP = args.unzip
LABELS = args.labels
PREPROCESSING = args.preprocessing
TRAIN_CNN = args.train_cnn
DATA_ZIPFILE = f'data/{args.file}'

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

params = {
    "N_TRAIN": 22,
    "N_TEST": 6,
    "N_FOLDS": 10,
    "TARGET_SIZE": 128,
    "AUGMENTATION_FACTOR": 40,  # Multiplying n_images by this factor with rotating and flipping
    "BATCH_SIZE": 20000,  # Number of sampled patches
    "THRESHOLD_DEFECTIVE": 0.1,  # Lower limit for normalized defective area in order to be classified 'defective'
    "THRESHOLD_NONDEFECTIVE": 0.01, # Upper limit for normalized defective area in order to be classified 'non_defective'
}

# Specify Directories where data is or is going to be
parent_dir, name, image_format = re.split('\.|\/', DATA_ZIPFILE)

dir_data = os.path.join(parent_dir, name)

dir_defective = os.path.join(dir_data, 'defective')
dir_defective_images = os.path.join(dir_defective, 'images')
dir_defective_annotations = os.path.join(dir_defective, 'annotations')
dir_defective_labels = os.path.join(dir_defective, 'labels')
dir_defective_json = os.path.join(dir_defective, 'json')

dir_nondefective = os.path.join(dir_data, 'non_defective')
dir_nondefective_images = os.path.join(dir_nondefective, 'images')
dir_nondefective_annotations = os.path.join(dir_nondefective, 'annotations')
dir_nondefective_labels = os.path.join(dir_nondefective, 'labels')

dir_folds = os.path.join(dir_data, f'n_train_{params["N_TRAIN"]}')
dir_output = os.path.join(f'output', name, f'n_train_{params["N_TRAIN"]}')

if UNZIP:
    if image_format == 'zip':
        # Unzip Data
        with ZipFile(DATA_ZIPFILE, 'r') as zip_ref:
            zip_ref.extractall(parent_dir)

if LABELS:
    # Get labels for non-defective images
    get_labels(
        dir_labels=dir_nondefective_labels,
        dir_annotated=dir_nondefective_images,
        no_label=True,
    )
    # Get labels for defective images
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

if PREPROCESSING:
    run(dir_data, dir_defective, dir_nondefective, dir_folds, params)

if TRAIN_CNN:
    train_cnn(dir_folds=dir_folds, output_dir=dir_output, n_folds=params["N_FOLDS"])



