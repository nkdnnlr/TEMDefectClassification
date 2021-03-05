import os
import shutil
import random
import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib.image as img
import numpy as np


def balance(dir1, dir2):
    """
    Balances two directories containing files by deleting from larger (by number of files).
    Get minimum number of files per directory, then randomly delete from larger until equal number.
    TODO: Write general for n directories.
    :param dir1: Path to first directory
    :param dir2: Path to first directory
    :return:
    """
    print("Balance directories")

    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    n1 = len(files1)
    n2 = len(files2)
    if n1 == n2:
        print("equal")
        return
    else:
        n_min = min(n1, n2)
        n_max = max(n1, n2)

        random.shuffle(files1)
        random.shuffle(files2)
        for file in files1:
            if file not in files1[0:n_min]:
                path_to_delete = os.path.join(dir1, file)
                os.remove(path_to_delete)
        for file in files2:
            if file not in files2[0:n_min]:
                path_to_delete = os.path.join(dir2, file)
                os.remove(path_to_delete)


def sort_folds_by_label(dir_target, n_folds, threshold_defective=0.1, threshold_nondefective=0.01, nolabels=False):
    """

    :param n_folds:
    :param threshold_defective:
    :param threshold_nondefective:
    :param nolabels:
    :return:
    """
    for fold in tqdm.trange(n_folds):
        print("Sort fold{} by label".format(fold))
        print("train")
        sort_by_label(dir_target + "/fold{}/train".format(str(fold)), threshold_defective=threshold_defective, threshold_nondefective=threshold_nondefective, nolabels=nolabels)
        print("test")
        sort_by_label(dir_target + "/fold{}/test".format(str(fold)), threshold_defective=threshold_defective, threshold_nondefective=threshold_nondefective, nolabels=nolabels)

def sort_folds_by_label_(dir_target, n_folds, threshold_defective=0.1, threshold_nondefective=0.01, nolabels=False):
    """

    :param n_folds:
    :param threshold_defective:
    :param threshold_nondefective:
    :param nolabels:
    :return:
    """
    for fold in tqdm.trange(n_folds):
        print("Sort fold{} by label".format(fold))
        print("train")
        sort_by_label_(dir_target + "/fold{}/train".format(str(fold)), threshold_defective=threshold_defective,
        threshold_nondefective=threshold_nondefective, nolabels=nolabels)
        print("test")
        sort_by_label_(dir_target + "/fold{}/test".format(str(fold)), threshold_defective=threshold_defective,
                       threshold_nondefective=threshold_nondefective, nolabels=nolabels)


def sort_by_label(dir_data, threshold_defective=0.1, threshold_nondefective=0.01, nolabels=False):
    """

    :param dir_data:
    :param threshold_defective:
    :param threshold_nondefective:
    :param nolabels:
    :return:
    """
    dir_images = os.path.join(dir_data, "patches", "images")
    dir_labels = os.path.join(dir_data, "patches", "labels")

    dir_nondefective_images = os.path.join(dir_data, "classes", "nondefective", "images")
    dir_nondefective_labels = os.path.join(dir_data, "classes", "nondefective", "labels")
    dir_defective_images = os.path.join(dir_data, "classes", "defective", "images")
    dir_defective_labels = os.path.join(dir_data, "classes", "defective", "labels")
    dir_neglecting_images = os.path.join(dir_data, "classes", "neglecting", "images")
    dir_neglecting_labels = os.path.join(dir_data, "classes", "neglecting", "labels")
    if os.path.exists(dir_nondefective_images) and os.path.isdir(
        dir_nondefective_images
    ):
        shutil.rmtree(dir_nondefective_images)
        shutil.rmtree(dir_nondefective_labels)
        shutil.rmtree(dir_defective_images)
        shutil.rmtree(dir_defective_labels)
    os.makedirs(dir_nondefective_images)
    os.makedirs(dir_nondefective_labels)
    os.makedirs(dir_defective_images)
    os.makedirs(dir_defective_labels)
    os.makedirs(dir_neglecting_images)
    os.makedirs(dir_neglecting_labels)

    files = [file for file in os.listdir(dir_labels) if file.endswith(".tif")]

    for file in files:
        file_label = os.path.join(dir_labels, file)
        file_image = os.path.join(dir_images, file)

        if nolabels:
            shutil.copy(file_image, dir_nondefective_images)
            shutil.copy(file_label, dir_nondefective_labels)

        else:
            label = img.imread(file_label)
            # image = img.imread(file_image)
            ratio = label.mean() / 255

            if ratio >= threshold_defective:
                shutil.copy(file_image, dir_defective_images)
                shutil.copy(file_label, dir_defective_labels)
            elif ratio <= threshold_nondefective:
                shutil.copy(file_image, dir_nondefective_images)
                shutil.copy(file_label, dir_nondefective_labels)
            else:
                shutil.copy(file_image, dir_neglecting_images)
                shutil.copy(file_label, dir_neglecting_labels)

def sort_by_label_(dir_data, threshold_defective=0.1, threshold_nondefective=0.01, nolabels=False):
    """

    :param dir_data:
    :param threshold_defective:
    :param threshold_nondefective:
    :param nolabels:
    :return:
    """
    dir_images = os.path.join(dir_data, "patches", "images")
    dir_labels = os.path.join(dir_data, "patches", "labels")

    dir_nondefective_images = os.path.join(dir_data, "classes", "nondefective", "images")
    dir_nondefective_labels = os.path.join(dir_data, "classes", "nondefective", "labels")
    dir_defective_images = os.path.join(dir_data, "classes", "defective", "images")
    dir_defective_labels = os.path.join(dir_data, "classes", "defective", "labels")
    dir_neglecting_images = os.path.join(dir_data, "classes", "neglecting", "images")
    dir_neglecting_labels = os.path.join(dir_data, "classes", "neglecting", "labels")
    if os.path.exists(dir_nondefective_images) and os.path.isdir(
        dir_nondefective_images
    ):
        shutil.rmtree(dir_nondefective_images)
        shutil.rmtree(dir_nondefective_labels)
        shutil.rmtree(dir_defective_images)
        shutil.rmtree(dir_defective_labels)
    os.makedirs(dir_nondefective_images)
    os.makedirs(dir_nondefective_labels)
    os.makedirs(dir_defective_images)
    os.makedirs(dir_defective_labels)
    os.makedirs(dir_neglecting_images)
    os.makedirs(dir_neglecting_labels)

    files = [file for file in os.listdir(dir_labels) if file.endswith(".tif")]

    for file in files:
        file_label = os.path.join(dir_labels, file)
        file_image = os.path.join(dir_images, file)

        if nolabels:
            shutil.copy(file_image, dir_nondefective_images)
            shutil.copy(file_label, dir_nondefective_labels)

        else:
            label = img.imread(file_label)

            unique, count = np.unique(label, return_counts = True)

            pixel_dict = {u: c for u, c in zip(unique, count)}

            r_nondef = 0
            r_def = 0
            r_blur = 0
            try:
                r_nondef = pixel_dict[0]/(128*128)
            except KeyError:
                pass
            try:
                r_def = pixel_dict[127]/(128*128)
            except KeyError:
                pass
            try:
                r_blur = pixel_dict[255]/(128*128)
            except KeyError:
                pass
            # print(r_nondef, r_def, r_blur)

            thr_def = 0.1
            thr_nondef = 0.01

            # Defective
            if r_blur > thr_def:
                shutil.copy(file_image, dir_defective_images)
                shutil.copy(file_label, dir_defective_labels)
            elif (r_def > thr_def) and (r_def < (1-thr_def)):
                shutil.copy(file_image, dir_defective_images)
                shutil.copy(file_label, dir_defective_labels)

            # Non-Defective
            elif (r_blur < thr_nondef) and ((r_def<thr_nondef) or (r_def>(1-thr_nondef))):
                shutil.copy(file_image, dir_nondefective_images)
                shutil.copy(file_label, dir_nondefective_labels)
            # Neglected
            else:
                shutil.copy(file_image, dir_neglecting_images)
                shutil.copy(file_label, dir_neglecting_labels)


def remove_redundant_dirs(dir_target, n_folds, *args):
    """
    Removes subdirectories specified in *args
    :param dir_target:
    :param args:
    :return:
    """
    # print(args)
    for fold in range(n_folds):
        dir_fold = os.path.join(dir_target, "fold{}".format(str(fold)))
        dir_train = os.path.join(dir_fold, "train")
        dir_test = os.path.join(dir_fold, "test")

        train_test = [dir_train, dir_test]
        for dir in train_test:
            for arg in args:
                path_to_delete = os.path.join(dir, arg)
                if os.path.exists(path_to_delete):
                    print("Deleting directory at {}".format(path_to_delete))
                    shutil.rmtree(path_to_delete, ignore_errors=True)
                else:
                    print("Directory at {} doesn't exist".format(path_to_delete))


def make_cnn_structure(dir_target, dir_output, fold=0, balanced=True):
    """
    Make folder structure that is suitable for CNN architecture. Balance dataset (by deleting from larger).
    For now, uses only defective images (but separates non-defective from defective patches)

    Receives target directory with the structure
    dir_target
            defective
                folds
                    fold{int}
                        train
                            classes
                                nondefective
                                    images
                                    ...
                                defective
                                    images
                                    ...
                            ...
                        test
                            classes
                                nondefective
                                    images
                                    ...
                                defective
                                    images
                                    ...
                            ...
                    ...
            non_defective
                folds
                    fold0
                        train
                            nondefective
                                images
                                ...
                            defective
                                images
                                ...
                            ...
                        test
                            nondefective
                                images
                                ...
                            defective
                                images
                                ...
                            ...
                    ...

    and creates output directory with the structure
    dir_output
        train
            non_defective
            defective
        test
            non_defective
            defective
    :param dir_target: target directory with above structure
    :param dir_output: output directory with above structure
    :param balanced: balance classes by deleting from larger
    :return:
    """
    print("Creating CNN structure for fold{}.".format(fold))
    target_fold = os.path.join(dir_target, "defective", "folds", "fold{}".format(fold))
    target_train_nondefective = os.path.join(target_fold, "train", "classes", "nondefective", "images")
    target_train_defective = os.path.join(target_fold, "train", "classes", "defective", "images")
    target_test_nondefective = os.path.join(target_fold, "test", "classes", "nondefective", "images")
    target_test_defective = os.path.join(target_fold, "test", "classes", "defective", "images")

    target_dirs = [target_train_nondefective, target_train_defective, target_test_nondefective, target_test_defective]

    for target_dir in target_dirs:
        # print(target_dir)
        print("HERE")
        cwd = os.getcwd()
        print(cwd)
        print(target_dir)
        assert os.path.isdir(target_dir)

    cnn_fold = os.path.join(dir_output, "fold{}".format(fold))
    cnn_train_nondefective = os.path.join(cnn_fold, "train", "non_defective")
    cnn_train_defective = os.path.join(cnn_fold, "train", "defective")
    cnn_test_nondefective = os.path.join(cnn_fold, "test", "non_defective")
    cnn_test_defective = os.path.join(cnn_fold, "test", "defective")

    cnn_dirs = [cnn_train_nondefective, cnn_train_defective, cnn_test_nondefective, cnn_test_defective]

    if os.path.exists(cnn_fold):
        shutil.rmtree(cnn_fold)

    for cnn_dir in cnn_dirs:
        if os.path.exists(cnn_dir):
            shutil.rmtree(cnn_dir)
        os.makedirs(cnn_dir)

    shutil.copy(os.path.join(target_fold, "train.txt"), os.path.join(cnn_fold, "train.txt"))
    shutil.copy(os.path.join(target_fold, "test.txt"), os.path.join(cnn_fold, "test.txt"))

    assert len(target_dirs) == len(cnn_dirs)
    for i in range(len(target_dirs)):
        target_dir = target_dirs[i]
        cnn_dir = cnn_dirs[i]
        files = [file for file in os.listdir(target_dir) if file.endswith(".tif")]
        for file in files:
            target_file = os.path.join(target_dir, file)
            shutil.copy(target_file, cnn_dir)

    if balanced:
        balance(cnn_train_nondefective, cnn_train_defective)
        balance(cnn_test_nondefective, cnn_test_defective)