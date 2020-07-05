import os
import sys
import argparse
import re
from zipfile import ZipFile

from src.utils.get_labelsmask import get_labels
from src.utils.run_preprocessing import run

parser = argparse.ArgumentParser()
# Data structure arguments
parser.add_argument("-file", "--file", type=str, default="cubic.zip",
                    help="Name of zipped data file.")
parser.add_argument("-unzip", "--unzip", type=bool, default=True,
                    help="Unzip?")
parser.add_argument("-labels", "--labels", type=bool, default=True,
                    help="Convert annotations to labels?")
parser.add_argument("-preprocessing", "--preprocessing", type=bool, default=True,
                    help="Preprocessing?.")
args = parser.parse_args()

UNZIP = args.unzip
LABELS = args.labels
PREPROCESSING = args.preprocessing
DATA_ZIPFILE = f'data/{args.file}'

dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

# # Overwrite DATA_ZIPFILE with parsing argument
# if len(sys.argv) == 2:
#     DATA_ZIPFILE = sys.argv[-1]

# Specify Directories where data is or is going to be
parent_dir, name, image_format = re.split('\.|\/', DATA_ZIPFILE)

dir_data = os.path.join(parent_dir, name)

dir_defective = os.path.join(dir_data, 'defective')
dir_defective_images = os.path.join(dir_defective, 'images')
dir_defective_annotations = os.path.join(dir_defective, 'annotations')
dir_defective_labels = os.path.join(dir_defective, 'labels')

dir_nondefective = os.path.join(dir_data, 'non_defective')
dir_nondefective_images = os.path.join(dir_nondefective, 'images')
dir_nondefective_annotations = os.path.join(dir_nondefective, 'annotations')
dir_nondefective_labels = os.path.join(dir_nondefective, 'labels')

dir_folds = os.path.join(dir_data, '6folds_128')

if UNZIP:
    if image_format == 'zip':
        # Unzip Data
        with ZipFile(DATA_ZIPFILE, 'r') as zip_ref:
            zip_ref.extractall(parent_dir)

if LABELS:
    # Get labels for non-defective images (should be fast)
    get_labels(
        dir_labels=dir_nondefective_labels,
        dir_annotated=dir_nondefective_annotations,
        no_label=True,
    )
    # Get labels for defective images (will take a while)
    get_labels(
        dir_labels=dir_defective_labels,
        dir_annotated=dir_defective_annotations,
        no_label=False,
    )

if PREPROCESSING:
    run(dir_data, dir_defective, dir_nondefective, dir_folds)





