import os
import sys
import shutil
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.preprocessing import preprocess_folder
from src.utils.augmentation import create_folds, augment_folds, randomcrop_folds, get_spectrum_for_cnn
from src.utils.modify_folders import sort_folds_by_label, sort_folds_by_label_, remove_redundant_dirs, \
    make_cnn_structure


def run(DIR_PARENT, DIR_DEFECTIVE, DIR_NONDEFECTIVE, DIR_FOLDS):
    # Define image patches
    TARGET_SIZE = 128
    # Define fold splitting settings
    KFOLD = 6
    N_SPLITS = 6
    # Define augmentation settings
    AUGMENTATION_FACTOR = 20  # Multiplying n_images by this factor with rotating and flipping
    BATCH_SIZE = 10000  # Number of sampled patches
    THRESHOLD_DEFECTIVE = 0.1  # Lower limit for normalized defective area in order to be classified 'defective'
    THRESHOLD_NONDEFECTIVE = 0.01  # Upper limit for normalized defective area in order to be classified 'non_defective'

    dirs = [DIR_DEFECTIVE]#, DIR_NONDEFECTIVE]
    for dir in dirs:
        dir_original = os.path.join(dir, "images")
        dir_preprocessed = os.path.join(dir, "preprocessed")
        dir_labels = os.path.join(dir, "labels")
        # dir_labels_blur = os.path.join(dir, "labels_blur")
        # dir_labels_symmetry = os.path.join(dir, "labels_symmetry")
        dir_target = os.path.join(dir, "folds")

        _nolabels = False
        if dir == DIR_NONDEFECTIVE:
            _nolabels = True

        # Preprocessing
        preprocess_folder(directory_original=dir_original, directory_preprocessed=dir_preprocessed)

        # Splitting into folds
        create_folds(dir_images=dir_preprocessed, dir_labels=dir_labels, dir_target=dir_target, kfold=KFOLD,
                      n_folds=N_SPLITS)

        # Augmenting
        print("AUGMENTING...")
        augment_folds(dir_data=dir_target, m=AUGMENTATION_FACTOR)
        print("RANDOM CROPPING...")
        randomcrop_folds(dir_data=dir_target, crop_target=TARGET_SIZE, batch_size=BATCH_SIZE, intensity_flip=True)

        # exit()
        # sort_folds_by_label(dir_target=dir_target, n_folds=N_SPLITS, threshold_defective=THRESHOLD_DEFECTIVE, threshold_nondefective=THRESHOLD_NONDEFECTIVE, nolabels=False)
        print("SORT FOLD BY LABEL...")
        sort_folds_by_label_(dir_target=dir_target, n_folds=N_SPLITS, threshold_defective=THRESHOLD_DEFECTIVE,
                             threshold_nondefective=THRESHOLD_NONDEFECTIVE, nolabels=False)

        # Deleting redundant files to save storage
        print("REMOVE REDUNDANT DIRS...")
        remove_redundant_dirs(dir_target, N_SPLITS, "augmented", "patches")

    # For each fold, make CNN structure
    for fold in tqdm.trange(N_SPLITS):
        make_cnn_structure(dir_target=DIR_PARENT,
                           dir_output=DIR_FOLDS,
                           fold=fold,
                           balanced=True)

    # Delete redundant
    for dir in dirs:
        dir_target = os.path.join(dir, "folds")
        if os.path.exists(dir_target):
            print("Deleting directory at {}".format(dir_target))
            shutil.rmtree(dir_target, ignore_errors=True)


if __name__ == '__main__':
    # Define directories
    DIR_PARENT = "../../data/cubic"
    DIR_DEFECTIVE = os.path.join(DIR_PARENT, "defective/")
    DIR_NONDEFECTIVE = os.path.join(DIR_PARENT, "non_defective/")
    DIR_FOLDS = os.path.join(DIR_PARENT, "6fold_128")
    run(DIR_PARENT, DIR_DEFECTIVE, DIR_NONDEFECTIVE, DIR_FOLDS)