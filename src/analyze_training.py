import os
import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.image as img
from keras.models import load_model
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from src.utils.helpers import get_image, get_patches
from src.utils.predict_class import predict
from src.utils.preprocessing import preprocess_image
import skimage

import utils.helpers as helpers

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compare_n_train(dir, output_dir):
    n_train = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    # fig, ax = plt.subplots(ncols=2)
    fig, ax = plt.subplots()

    all_val_accs = []
    for n in n_train:
        print(n)
        subdir = os.path.join(dir, f"n_train_{n}")


        val_accs = []
        for subsubsir in sorted(os.listdir(subdir)):
            if not os.path.isdir(os.path.join(subdir, subsubsir)):
                continue

            file_path = os.path.join(subdir, subsubsir, "trainHistoryDict.json")
            # print(file_path)

            # print("Hu")

            with open(file_path, 'r') as f:
                string = str(json.load(f))
                string = string.replace('\'', '\"')
                data_dict = json.loads(string)
            # print(data_dict.keys())
            loss = np.array(data_dict['loss'])
            acc = np.array(data_dict['accuracy'])
            val_loss = np.array(data_dict['val_loss'])
            val_acc = np.array(data_dict['val_accuracy'])

            # print(len(val_acc))
            # if len(val_acc) != 20:
            #     continue
            val_accs.append(val_acc[19])


        mean_val_acc = np.mean(val_accs)
        std_val_acc = np.std(val_accs)

        all_val_accs.append(val_accs)

    positions = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    ax.boxplot(all_val_accs, widths=0.5, patch_artist=True, showmeans=False, boxprops=dict(facecolor='indigo',
                                                                                           color='indigo'),
               medianprops=dict(color='yellow'),
               flierprops=dict(marker='o', markerfacecolor='w', markersize=4,
                  linestyle='none', markeredgecolor='k'),
               )

    # xticklabels = range(2, 23, 2)
    ax.set_xticklabels(positions)
    ax.set_ylabel("Validation Accuracy")
    ax.set_xlabel("# Training Images")
    plt.grid(axis='y')
    # ax.set_ylim([0,1])

    plt.savefig(os.path.join(output_dir, 'valacc_vs_ntrain_boxplot.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'valacc_vs_ntrain_boxplot.svg'))
    plt.show()

    exit()
            # print(loss)
            # print(acc)
            # print(val_loss)
            # print(val_acc)
            # exit()

# compare_n_train()


def compare_train_test(dir, output_dir):
    # print(dir)
    reds = iter(cm.Reds(np.linspace(0, 1, 10)))
    purples1 = iter(cm.Purples(np.linspace(0.2, 1, 10)))
    purples2 = iter(cm.Purples(np.linspace(0.2, 1, 10)))
    purples3 = iter(cm.Purples(np.linspace(0.2, 1, 10)))
    purples4 = iter(cm.Purples(np.linspace(0.2, 1, 10)))

    misc = iter(cm.jet(np.linspace(0, 1, 10)))

    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
    axes = axes.reshape(-1)

    losses = None
    val_losses = None
    accuracies = None
    val_accuracies = None

    fold = 1
    print(dir)
    for subdir in sorted(os.listdir(dir)):
        print(subdir)
        # print("    " + subdir)
        for file in sorted(os.listdir(os.path.join(dir, subdir))):
            if file.endswith(".json"):
                file_path = os.path.join(dir, subdir, file)
                # print(file_path)
                with open(file_path, 'r') as f:
                    string = str(json.load(f))
                    string = string.replace('\'', '\"')
                    data_dict = json.loads(string)
                # print(data_dict.keys())
                loss = np.array(data_dict['loss'])
                acc = np.array(data_dict['accuracy'])
                val_loss = np.array(data_dict['val_loss'])
                val_acc = np.array(data_dict['val_accuracy'])

                # print(data_dict)
                # print(loss.shape)
                # print(acc.shape)
                # print(val_loss.shape)
                # print(val_acc.shape)
                # exit()
                # print(losses)
                # print(loss)
                if losses is None:
                    losses = np.array(loss)
                else:
                    losses += np.array(loss)

                if val_losses is None:
                    val_losses = np.array(val_loss)
                else:
                    val_losses += np.array(val_loss)

                if accuracies is None:
                    accuracies = np.array(acc)
                else:
                    accuracies += np.array(acc)

                if val_accuracies is None:
                    val_accuracies = np.array(val_acc)
                else:
                    val_accuracies += np.array(val_acc)


                # print(val_acc[-1])

                axes[0].plot(loss, label=None, color=next(purples1))
                # axes[0].set_ylim([0.45, 1.02])
                handles, labels = axes[0].get_legend_handles_labels()
                # axes[0].text(0, 1.03, "A", transform=axes[0].transAxes,
                #         size=16, weight='bold')
                axes[0].set_ylabel("CCE-Loss") #Categorical Crossentropy-Loss
                # axes[0].set_xlabel("Epochs")
                axes[0].set_title("Train Set (22 images)")


                axes[1].plot(val_loss, label=None, color=next(purples2))
                # axes[1].set_ylim([0.45, 1.02])
                handles, labels = axes[1].get_legend_handles_labels()
                # axes[1].text(0, 1.03, "B", transform=axes[1].transAxes,
                #         size=16, weight='bold')
                # axes[1].set_ylabel("Accuracy")
                # axes[1].set_xlabel("Epochs")
                # axes[1].set_yticks([])
                axes[1].set_title("Validation Set (6 images)")

                axes[2].plot(acc, label=None, color=next(purples3))
                axes[2].set_ylim([0.65, 1.02])
                handles, labels = axes[0].get_legend_handles_labels()
                # axes[2].text(0, 1.03, "A", transform=axes[0].transAxes,
                #         size=16, weight='bold')
                axes[2].set_ylabel("Accuracy")
                axes[2].set_xlabel("Epochs")
                # axes[2].set_title("Train")


                axes[3].plot(val_acc, label=fold-1, color=next(purples4))
                axes[3].set_ylim([0.65, 1.02])
                handles, labels = axes[1].get_legend_handles_labels()
                # axes[3].text(0, 1.03, "B", transform=axes[1].transAxes,
                #         size=16, weight='bold')
                # axes[1].set_ylabel("Accuracy")
                axes[3].set_xlabel("Epochs")
                # axes[3].set_yticks([])
                # axes[3].set_title("Validation")

                # print(fold)
                print(val_acc[-1])
                fold += 1
        if fold >= 10 + 1:
            mean_valloss = val_losses / (fold - 1)
            mean_loss = losses / (fold - 1)
            mean_valacc = val_accuracies / (fold - 1)
            mean_acc = accuracies / (fold - 1)
            print("MEAN ACC (train, test)")
            print(mean_acc[-1])
            print(mean_valacc[-1])
            axes[0].plot(mean_loss, 'r', label="mean", linewidth=1.5)
            axes[1].plot(mean_valloss, 'r', label="mean", linewidth=1.5)
            axes[2].plot(mean_acc, 'r', label="mean", linewidth=1.5)
            axes[3].plot(mean_valacc, 'r', label="mean", linewidth=1.5)

        handles, labels = axes[0].get_legend_handles_labels()
        lines, labels = fig.axes[-1].get_legend_handles_labels()
            # handles, labels = axes[1].get_legend_handles_labels()

            # plt.legend()
            #     break
    # fig.legend(bbox_to_anchor=(1, 0), loc="lower left", mode="expand", ncol=2
    # print(handles)
    # print(labels)

    axes[0].legend(lines, [int(label)+1 if len(label) == 1 else label for label in labels], loc="upper right",
                   ncol=3)

    # plt.legend()
    # fig.legend(handles, labels, loc='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "traintest_analysis.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "traintest_analysis.svg"))
    plt.show()
    # print("Done")


def compare_many(parent_dir, output_dir, subplottitles, set='val'):
    # plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(3, 2)#, figsize=(8, 8))
    for nr, ax in enumerate(axes.flatten()):
        dir = sorted(os.listdir(parent_dir))[nr]
        print(dir)
        reds = iter(cm.Reds(np.linspace(0, 1, 6)))
        purples = iter(cm.Purples(np.linspace(0.2, 1, 6)))
        misc = iter(cm.jet(np.linspace(0, 1, 6)))

        accuracies = None

        fold = 1
        for subdir in sorted(os.listdir(os.path.join(parent_dir, dir))):
            # print("    " + subdir)
            for file in sorted(os.listdir(os.path.join(parent_dir, dir, subdir))):
                if file.endswith(".json"):
                    file_path = os.path.join(parent_dir, dir, subdir, file)
                    # print(file_path)
                    with open(file_path, 'r') as f:
                        string = str(json.load(f))
                        string = string.replace('\'', '\"')
                        data_dict = json.loads(string)
                    # print(data_dict.keys())
                    loss = data_dict['loss']
                    acc = data_dict['acc']
                    val_lossl\
                        = data_dict['val_loss']
                    if set == 'train':
                        val_acc = data_dict['acc']

                    else:
                        val_acc = data_dict['val_acc']

                    if accuracies is None:
                        accuracies = np.array(val_acc)
                    else:
                        accuracies += np.array(val_acc)

                    ax.plot(val_acc, label=fold, color=next(purples))
                    # ax.plot(acc, label=fold, color=next(purples))
                    ax.set_ylim([0.4,1.02])
                    # ax.legend()
                    handles, labels = ax.get_legend_handles_labels()

                    if nr in [0, 2, 4]:
                        ax.set_ylabel("Accuray")
                    else:
                        ax.set_yticks([])

                    if nr in [4, 5]:
                        ax.set_xlabel("Epochs")

                    # ax.set_title(str(dir))
                    # ax.text(0, 1.03, subplottitles[nr], transform=ax.transAxes,
                    #          size=16, weight='bold')
                    fold += 1
                    print(fold)
            if fold >= 6+1:
                mean_acc = accuracies/(fold-1)
                print(mean_acc[-1])
                ax.plot(mean_acc, 'r', label="mean", linewidth=1.5)
                handles, labels = ax.get_legend_handles_labels()
                break
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "{}accuracy_128vs224.png".format(set)))
    plt.show()

def test_on_nondefective():
    fig, axes = plt.subplots(3, 2)#, figsize=(8, 8))
    for nr, ax in enumerate(axes.flatten()):
        dir = sorted(os.listdir(parent_dir))[nr]

def test_accuracy_nondefective(path_model, images_nondefective):
    # NONDEFECTIVE
    print(path_model)
    assert os.path.exists(path_model)
    model = load_model(os.path.join(path_model, "model.h5"))

    target_size = 128
    sum_all = 0
    sum_defective = 0
    accuracies = []
    for file in os.listdir(images_nondefective):
        if file.endswith(".tif"):
            # print(file)
            # fig, ax = plt.subplots(figsize=(8, 8))
            # LOAD IMAGE AND PREPROCESS
            # name = file[:-4]
            file = os.path.join(images_nondefective, file)
            original = get_image(file)

            image = skimage.filters.gaussian(original, 2) - skimage.filters.gaussian(original,
                                                                                     3)  # Looks quite much
            # like previous setting
            image = np.interp(image, (image.min(), image.max()), (0, 254))

            STEP_SIZE = target_size
            patches = get_patches(image, target_size=target_size, step=STEP_SIZE)

            # PREDICT DEFECT ON PATCHES WITH CNN, THEN ACCUMULATE TO IMAGE
            # print("Predicting {} patches...".format(len(patches)))
            classes = predict(patches, model)[:, 0].round().astype(int)  # One-hot encoded
            n_patches = classes.shape[0]
            sum_all += n_patches
            sum_defective += len(classes[classes == 1])
        print("sum_all: ", sum_all)
        print("sum_defective: ", sum_defective)
        print("accuracy: ", 1. - sum_defective / sum_all)
        accuracies.append(1. - sum_defective / sum_all)
    print(accuracies)

def test_accuracy(path_model, images_nondefective, images_defective, labels_defective, output_dir):
    target_size = 128
    thr_def = 0.05

    print(path_model)
    assert os.path.exists(path_model)
    model = load_model(os.path.join(path_model, "model.h5"))

    train_images = []
    val_images = []
    with open(os.path.join(path_model, 'train.txt'), 'r') as f:
        for line in f:
            w = line.strip().split(sep=": ")[1]
            train_images.append(w)
    with open(os.path.join(path_model, 'test.txt'), 'r') as f:
        for line in f:
            w = line.strip().split(sep=": ")[1]
            val_images.append(w)

    # print(train_images)
    # print(val_images)
    # exit()
        # train_images =



    # exit()
    # DEFECTIVE
    # sum_all = 0
    # sum_defective = 0
    # accuracies = []

    # train_acc = []
    # val_acc = []
    # test_acc = []
    #
    # train_specificity = []
    # val_specificity = []
    # test_specificity = []
    #
    # train_precision = []
    # val_precision = []
    # test_precision = []
    #
    # train_recall = []
    # val_recall = []
    # test_recall = []

    train_tp = []
    train_tn = []
    train_fp = []
    train_fn = []

    val_tp = []
    val_tn = []
    val_fp = []
    val_fn = []

    test_tp = []
    test_tn = []
    test_fp = []
    test_fn = []




    # train_confm = []
    # val_confm = []
    # test_confm = []


    n_train = 0
    n_val = 0
    n_test = 0
    for file in os.listdir(images_defective):
        if not file.endswith(".tif"):
            continue
        # print(file)



        # print(state)
#         continue
# print(n_train, n_val, n_test)
# exit()
# if False:
#     if False:
        # fig, ax = plt.subplots(figsize=(8, 8))
        # LOAD IMAGE AND PREPROCESS
        # name = file[:-4]
        file_label = os.path.join(labels_defective, file)
        file_image = os.path.join(images_defective, file)

        original = get_image(file_image)
        # label =  get_image(file_label)
        label = img.imread(file_label)

        unique, count = np.unique(label, return_counts = True)




        # continue

        # original = get_image(file_image)

        image = skimage.filters.gaussian(original, 2) - skimage.filters.gaussian(original,
                                                                                 3)  # Looks quite much
        # like previous setting
        image = np.interp(image, (image.min(), image.max()), (0, 254))

        STEP_SIZE = target_size
        patches = get_patches(image, target_size=target_size, step=STEP_SIZE)
        patches_label = get_patches(label, target_size=target_size, step=STEP_SIZE)
        true_claases = helpers.labels2classes(patches_label, thr_def=thr_def)
        predicted_classes = predict(patches, model)[:, 0].round().astype(int)  # One-hot encoded

        # true_classes = np.array(patches_label)[np.array(patches_label)>0]
        # print(np.array(patches_label)/)
        # print(predicted_classes)
        #
        # print(patches_label.shape)
        # print(predicted_classes.shape)
        #
        # exit()

        score_map_true = helpers.map_scores(image, scores=true_claases, target_size=target_size,
                                          step_size=STEP_SIZE)
        score_map_predicted = helpers.map_scores(image, scores=predicted_classes, target_size=target_size,
                                        step_size=STEP_SIZE)


        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for patch, patch_label, pclass in zip(patches, patches_label, predicted_classes):
            unique, count = np.unique(patch_label, return_counts=True)

            pixel_dict = {u: c for u, c in zip(unique, count)}

            r_nondef = 0
            r_def = 0
            r_blur = 0
            try:
                r_nondef = pixel_dict[0] / (128 * 128)
            except KeyError:
                pass
            try:
                r_def = pixel_dict[127] / (128 * 128)
            except KeyError:
                pass
            try:
                r_blur = pixel_dict[255] / (128 * 128)
            except KeyError:
                pass
            # print(r_nondef, r_def, r_blur)


            # Defective
            if (r_blur > thr_def) or ((r_def > thr_def) and (r_def < (1 - thr_def))):
                if pclass == 1:
                    tp += 1
                elif pclass == 0:
                    fn += 1
                else:
                    print("something is wrong")

            else:
                if pclass == 1:
                    fp += 1
                elif pclass == 0:
                    tn += 1
                else:
                    print("something is wrong")


        confm = np.array([tp, tn, fp, fn]) / 64
        # print(confm)
        # acc = (tp + tn)/ (tp+tn+fp+fn)
        # specificity = tn/(tn+fp)
        # precision = tp/(tp+fp)
        # recall = tp/(tp+fn)

        if file in train_images:
            state = 'train'
            n_train += 1

            train_tp.append(tp)
            train_tn.append(tn)
            train_fp.append(fp)
            train_fn.append(fn)

            # train_acc.append(acc)
            # train_specificity.append(specificity)
            # train_precision.append(precision)
            # train_recall.append(recall)


            # train_confm.append(confm)
        elif file in val_images:
            state = 'val'
            n_val += 1
            # val_acc.append(acc)
            # val_specificity.append(specificity)
            # val_precision.append(precision)
            # val_recall.append(recall)

            val_tp.append(tp)
            val_tn.append(tn)
            val_fp.append(fp)
            val_fn.append(fn)


            # val_confm.append(confm)
        else:
            state = 'test'
            n_test += 1
            # test_acc.append(acc)
            # test_specificity.append(specificity)
            # test_precision.append(precision)
            # test_recall.append(recall)


            val_tp.append(tp)
            val_tn.append(tn)
            val_fp.append(fp)
            val_fn.append(fn)


            test_tp.append(tp)
            test_tn.append(tn)
            test_fp.append(fp)
            test_fn.append(fn)


            # test_confm.append(confm)

        # print(state)
        # continue
        # if n_train == 2:
        #     break

    train_tp = np.sum(train_tp)
    train_tn = np.sum(train_tn)
    train_fp = np.sum(train_fp)
    train_fn = np.sum(train_fn)

    val_tp = np.sum(val_tp)
    val_tn = np.sum(val_tn)
    val_fp = np.sum(val_fp)
    val_fn = np.sum(val_fn)

    test_tp = np.sum(test_tp)
    test_tn = np.sum(test_tn)
    test_fp = np.sum(test_fp)
    test_fn = np.sum(test_fn)


    train_acc = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
    train_specificity = train_tn / (train_tn + train_fp)
    train_precision = train_tp / (train_tp + train_fp)
    train_recall = train_tp / (train_tp + train_fn)

    val_acc = (val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn)
    val_specificity = val_tn / (val_tn + val_fp)
    val_precision = val_tp / (val_tp + val_fp)
    val_recall = val_tp / (val_tp + val_fn)

    test_acc = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)
    test_specificity = test_tn / (test_tn + test_fp)
    test_precision = test_tp / (test_tp + test_fp)
    test_recall = test_tp / (test_tp + test_fn)


    result_summary = [train_acc, train_specificity, train_precision, train_recall, val_acc, val_specificity,
                      val_precision, val_recall, test_acc, test_specificity, test_precision, test_recall]

    print("[train_acc, train_specificity, train_precision, train_recall, val_acc, val_specificity, val_precision, val_recall, test_acc, test_specificity, test_precision, test_recall]")
    print(result_summary)
    return result_summary

    # train_acc_mean = np.mean(train_acc)
    # train_acc_std = np.std(train_acc)
    # train_specificity_mean = np.mean(train_specificity)
    # train_specificity_std = np.std(train_specificity)
    # train_precision_mean = np.mean(train_precision)
    # train_precision_std = np.std(train_precision)
    # train_recall_mean = np.mean(train_recall)
    # train_recall_std = np.std(train_recall)
    #
    # val_acc_mean = np.mean(val_acc)
    # val_acc_std = np.std(val_acc)
    # val_specificity_mean = np.mean(val_specificity)
    # val_specificity_std = np.std(val_specificity)
    # val_precision_mean = np.mean(val_precision)
    # val_precision_std = np.std(val_precision)
    # val_recall_mean = np.mean(val_recall)
    # val_recall_std = np.std(val_recall)
    #
    # test_acc_mean = np.mean(test_acc)
    # test_acc_std = np.std(test_acc)
    # test_specificity_mean = np.mean(test_specificity)
    # test_specificity_std = np.std(test_specificity)
    # test_precision_mean = np.mean(test_precision)
    # test_precision_std = np.std(test_precision)
    # test_recall_mean = np.mean(test_recall)
    # test_recall_std = np.std(test_recall)
    #
    # result_summary = [train_acc_mean, train_acc_std, train_specificity_mean, train_specificity_std, train_precision_mean, train_precision_std,
    #                   train_recall_mean, train_recall_std, val_acc_mean, val_acc_std, val_specificity_mean, val_specificity_std,
    #                   val_precision_mean, val_precision_std, val_recall_mean, val_recall_std, test_acc_mean, test_acc_std,
    #                   test_specificity_mean, test_specificity_std, test_precision_mean, test_precision_std, test_recall_mean, test_recall_std]
    # print(train_tp, train_tn, train_fp, train_fn, val_tp, val_tn, val_fp, val_fn, test_tp, test_tn, test_fp, test_fn)
    # print(result_summary)
    # return result_summary
    #
    # print("")
    # print(f"train: {np.mean(train_acc)} pm {np.std(train_acc)}")
    # print(f"{np.mean(train_confm, axis=0)} pm {np.std(train_confm, axis=0)}")
    # print("")
    #
    # print(f"val: {np.mean(val_acc)} pm {np.std(val_acc)}")
    # print(f"{np.mean(val_confm, axis=0)} pm {np.std(val_confm, axis=0)}")
    # print("")
    #
    # print(f"test: {np.mean(test_acc)} pm {np.std(test_acc)}")
    # print(f"{np.mean(test_confm, axis=0)} pm {np.std(test_confm, axis=0)}")
    # print("")
    #
    # return np.mean(train_acc), np.std(train_acc), np.mean(train_confm, axis=0), np.std(train_confm, axis=0), \
    #        np.mean(val_acc), np.std(val_acc), np.mean(val_confm, axis=0), np.std(val_confm, axis=0), \
    #        np.mean(test_acc), np.std(test_acc), np.mean(test_confm, axis=0), np.std(test_confm, axis=0)
    # continue
    # exit()
            # continue

            #
            # exit()
            # # PREDICT DEFECT ON PATCHES WITH CNN, THEN ACCUMULATE TO IMAGE
            # print("Predicting {} patches...".format(len(patches)))
            # classes = predict(patches, model)[:, 0].round().astype(int)  # One-hot encoded
            # n_patches = classes.shape[0]
        #     sum_all += n_patches
        #     sum_defective += len(classes[classes == 1])
        # print("sum_all: ", sum_all)
        # print("sum_defective: ", sum_defective)
        # print("accuracy: ", 1. - sum_defective / sum_all)
        # accuracies.append(1. - sum_defective / sum_all)

    # print(accuracies)



def test_accuracy_nondefective(parent_dir_models, dir_images, output_dir=None, target_size=128):
    purples = cm.Purples(np.linspace(0.2, 1, 10))
    nrs = range(10)
    accuracies = []
    for nr in nrs:
        path_model = os.path.join(parent_dir_models, sorted(os.listdir(parent_dir_models))[nr],  "model.h5")
        print(path_model)
        assert os.path.exists(path_model)
        model = load_model(path_model)

        sum_all = 0
        sum_defective = 0
        for file in os.listdir(dir_images):
            if file.endswith(".tif"):
                fig, ax = plt.subplots(figsize=(8, 8))
                # LOAD IMAGE AND PREPROCESS
                name = file[:-4]
                file = os.path.join(dir_images, file)
                original = get_image(file)
                # image = preprocess_image(original,
                #                          lowpass_filter="gaussian", lowpass_kernel_size=5,
                #                          highpass_filter="laplacian", highpass_kernel_size=19,
                #                          rescale=False)

                image = skimage.filters.gaussian(original, 2) - skimage.filters.gaussian(original,
                                                                                         3)  # Looks quite much
                # like previous setting
                image = np.interp(image, (image.min(), image.max()), (0, 254))

                # CROP IN PATCHES
                STEP_SIZE = target_size
                patches = get_patches(image, target_size=target_size, step=STEP_SIZE)

                # PREDICT DEFECT ON PATCHES WITH CNN, THEN ACCUMULATE TO IMAGE
                print("Predicting {} patches...".format(len(patches)))
                classes = predict(patches, model)[:, 0].round().astype(int) # One-hot encoded
                n_patches = classes.shape[0]
                sum_all += n_patches
                sum_defective += len(classes[classes == 1])
        print("sum_all: ", sum_all)
        print("sum_defective: ", sum_defective)
        print("accuracy: ", 1.-sum_defective/sum_all)
        accuracies.append(1.-sum_defective/sum_all)
    print(accuracies)
    x = range(10)
    # accuracies = [0.9988839285714286, 1.0, 0.9720982142857143, 0.9994419642857143, 0.9955357142857143, 1.0]
    mean = np.mean(accuracies)
    fig, ax = plt.subplots(figsize=(3,3))
    plt.bar(x, accuracies, width=1, color=purples, edgecolor='black')
    print(mean)
    # plt.plot(np.arange(-1, 7)+0.5, [mean]*8, 'r')
    # plt.ylim([0.97, 1.00])
    # plt.xlim([-0.5, 5.5])
    # plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    # plt.yticks([0.97, 0.98, 0.99, mean, 1.00], [0.97, 0.98, 0.99, str(round(mean, 3)), 1.00])
    # ax.get_yticklabels()[3].set_color("red")
    plt.xlabel("Split Nr.")
    plt.ylabel("Accuracy")
    plt.title("Test")

    ax.text(0, 1.03, "C", transform=ax.transAxes,
                 size=16, weight='bold')

    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "test_accuracy_nondef_mean.pdf"))
    plt.show()
    # plt.plot(accuracies, 'ro')
    print("Done")



if __name__ == '__main__':
    output_dir = "../output/test/fold6split6_0p1_0p1"
    output_dir = "../output/all_data/n_train_8"
    output_dir = "../output/all_data/test"
    # parent_dir = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/models/vgg16_old/"
    # titles = ["A1T", "B1T", "A2T", "B2T", "A3T", "B3T"]
    # compare_many(parent_dir, output_dir, titles, set='train')
    # titles = ["A1V", "B1V", "A2V", "B2V", "A3V", "B3V"]
    # compare_many(parent_dir, output_dir, titles, set='val')

    if True:
        models_dir = "../models/all_data/n_train_16/newtraining"
        models_dir = "../models/all_data2"
        # model_dir = "../models/all_data/n_train_16/newtraining/fold2_20210317-233638"
        dir_images_nondefective = "../data/all_data/non_defective/images"
        dir_images_defective = "../data/all_data/defective/images"
        dir_labels_defective = "../data/all_data/defective/labels"

        test_accuracy_nondefective(models_dir, dir_images_nondefective)

        exit()
        all_train_acc = []
        all_val_acc = []
        all_test_acc = []

        all_train_acc_std = []
        all_val_acc_std = []
        all_test_acc_std = []

        all_train_confm = []
        all_val_confm = []
        all_test_confm = []

        all_train_confm_std = []
        all_val_confm_std = []
        all_test_confm_std = []

        all_results = []
        for model_dir in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, model_dir)
            results = test_accuracy(model_dir, dir_images_nondefective, dir_images_defective, dir_labels_defective,
                                  output_dir)

            all_results.append(results)

            # break
            continue

        print("Summary")
        print(np.mean(all_results, axis=0))
        print(np.std(all_results, axis=0))
        exit()

        #
        #
        # train_acc_mean, train_acc_std, train_confm_mean, train_confm_std, val_acc_mean, val_acc_std, \
        # val_confm_mean, val_confm_std, test_acc_mean, test_acc_std, test_confm_mean, test_confm_std = results
        #
        # all_train_acc.append(train_acc_mean)
        # all_train_acc_std.append(train_acc_std)
        # all_val_acc.append(val_acc_mean)
        # all_val_acc_std.append(val_acc_std)
        # all_test_acc.append(test_acc_mean)
        # all_test_acc_std.append(test_acc_std)
        #
        # all_train_confm.append(train_confm_mean)
        # all_train_confm_std.append(train_confm_std)
        # all_val_confm.append(val_confm_mean)
        # all_val_confm_std.append(val_confm_std)
        # all_test_confm.append(test_confm_mean)
        # all_test_confm_std.append(test_confm_std)
        #
        # break

    # print("summary")
    # print("all_train_acc_mean: ", np.mean(all_train_acc))
    # print("all_train_acc_std: ", np.mean(all_train_acc_std))
    # print("all_val_acc_mean: ", np.mean(all_val_acc))
    # print("all_val_acc_std: ", np.mean(all_val_acc_std))
    # print("all_test_acc_mean: ", np.mean(all_test_acc))
    # print("all_test_acc_std: ", np.mean(all_test_acc_std))
    #
    # all_train_confm = np.mean(all_train_confm, axis=0)
    # all_train_confm_std = np.mean(all_train_confm_std, axis=0)
    # all_val_confm = np.mean(all_val_confm, axis=0)
    # all_val_confm_std = np.mean(all_val_confm_std, axis=0)
    # all_test_confm = np.mean(all_test_confm, axis=0)
    # all_test_confm_std = np.mean(all_test_confm_std, axis=0)
    #
    # print("train")
    # print("specificity:", )
    #
    #
    # exit()

    if False:
        model_dir = "../models/all_data/n_train_16/newtraining"
        dir = "../data/all_data/non_defective/images"
        test_accuracy_nondefective(parent_dir_models=model_dir, dir_images=dir, output_dir=output_dir, target_size=128)
        exit()

    if True:
        # TRAIN TEST ANALYSIS
        dir = "../output/all_data2/n_train_22"
        compare_train_test(dir, output_dir)
        exit()
    if True:
        # BOXPLOT
        dir = "../output/all_data2"
        compare_n_train(dir, output_dir)
        exit()
    # dir = "../output/test/fold6split6_0p1_0p1"
    dir = "../output/all_data/n_train_8"
    compare_train_test(dir, output_dir)
    exit()

    # Test
    defective_dir = "../../data/cubic/defective/images/"
    nondefective_dir = "../../data/cubic/non_defective/images"
    TARGET_SIZE = 128
    parent_dir_models = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/models/finetuned/"
    output_dir = "../../output/pipeline/test"
    test_accuracy_nondefective(parent_dir_models, nondefective_dir, output_dir, target_size=TARGET_SIZE)


