import os
import json

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from keras.models import load_model

from src.utils.helpers import get_image, get_patches
from src.utils.predict_class import predict
from src.utils.preprocessing import preprocess_image




def compare_train_test(dir, output_dir):
    # print(dir)
    reds = iter(cm.Reds(np.linspace(0, 1, 6)))
    purples1 = iter(cm.Purples(np.linspace(0.2, 1, 6)))
    purples2 = iter(cm.Purples(np.linspace(0.2, 1, 6)))

    misc = iter(cm.jet(np.linspace(0, 1, 6)))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    accuracies = None
    val_accuracies = None

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
                val_lossl \
                    = data_dict['val_loss']
                val_acc = data_dict['val_acc']

                if accuracies is None:
                    accuracies = np.array(acc)
                else:
                    accuracies += np.array(acc)

                if val_accuracies is None:
                    val_accuracies = np.array(val_acc)
                else:
                    val_accuracies += np.array(val_acc)
                print(val_acc[-1])

                axes[0].plot(acc, label=fold, color=next(purples1))
                axes[0].set_ylim([0.45, 1.02])
                handles, labels = axes[0].get_legend_handles_labels()
                axes[0].text(0, 1.03, "A", transform=axes[0].transAxes,
                        size=16, weight='bold')
                axes[0].set_ylabel("Accuracy")
                axes[0].set_xlabel("Epochs")
                axes[0].set_title("Train")


                axes[1].plot(val_acc, label=fold, color=next(purples2))
                axes[1].set_ylim([0.45, 1.02])
                handles, labels = axes[1].get_legend_handles_labels()
                axes[1].text(0, 1.03, "B", transform=axes[1].transAxes,
                        size=16, weight='bold')
                # axes[1].set_ylabel("Accuracy")
                axes[1].set_xlabel("Epochs")
                axes[1].set_yticks([])
                axes[1].set_title("Validation")

                print(fold)
                fold += 1
        if fold >= 6 + 1:
            mean_valacc = val_accuracies / (fold - 1)
            mean_acc = accuracies / (fold - 1)
            print("NOW")
            print(mean_acc[-1])
            print(mean_valacc[-1])
            axes[0].plot(mean_acc, 'r', label="mean", linewidth=1.5)
            axes[1].plot(mean_valacc, 'r', label="mean", linewidth=1.5)

            handles, labels = axes[0].get_legend_handles_labels()
            handles, labels = axes[1].get_legend_handles_labels()
            break
    fig.legend(handles, labels, loc='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acc_vs_valacc.png"))
    plt.show()


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
                    ax.text(0, 1.03, subplottitles[nr], transform=ax.transAxes,
                             size=16, weight='bold')
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


def test_accuracy_nondefective(parent_dir_models, dir_images, output_dir, target_size=128):
    purples = cm.Purples(np.linspace(0.2, 1, 6))
    nrs = range(6)
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
                image = preprocess_image(original,
                                         lowpass_filter="gaussian", lowpass_kernel_size=5,
                                         highpass_filter="laplacian", highpass_kernel_size=19,
                                         rescale=False)
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
    x = range(6)
    accuracies = [0.9988839285714286, 1.0, 0.9720982142857143, 0.9994419642857143, 0.9955357142857143, 1.0]
    mean = np.mean(accuracies)
    fig, ax = plt.subplots(figsize=(3,3))
    plt.bar(x, accuracies, width=1, color=purples, edgecolor='black')
    print(mean)
    plt.plot(np.arange(-1, 7)+0.5, [mean]*8, 'r')
    plt.ylim([0.97, 1.00])
    plt.xlim([-0.5, 5.5])
    plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    plt.yticks([0.97, 0.98, 0.99, mean, 1.00], [0.97, 0.98, 0.99, str(round(mean, 3)), 1.00])
    ax.get_yticklabels()[3].set_color("red")
    plt.xlabel("Split Nr.")
    plt.ylabel("Accuracy")
    plt.title("Test")

    ax.text(0, 1.03, "C", transform=ax.transAxes,
                 size=16, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_accuracy_nondef_mean.pdf"))
    plt.show()
    # plt.plot(accuracies, 'ro')
    print("Done")



if __name__ == '__main__':
    output_dir = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/output/training/"
    parent_dir = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/models/vgg16_old/"
    titles = ["A1T", "B1T", "A2T", "B2T", "A3T", "B3T"]
    compare_many(parent_dir, output_dir, titles, set='train')
    titles = ["A1V", "B1V", "A2V", "B2V", "A3V", "B3V"]
    compare_many(parent_dir, output_dir, titles, set='val')

    dir = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/models/finetuned/"
    compare_train_test(dir, output_dir)

    # Test
    defective_dir = "../../data/cubic/defective/images/"
    nondefective_dir = "../../data/cubic/non_defective/images"
    TARGET_SIZE = 128
    parent_dir_models = "/home/nik/Documents/Defect Classification/TEMDefectAnalysis/models/finetuned/"
    output_dir = "../../output/pipeline/test"
    test_accuracy_nondefective(parent_dir_models, nondefective_dir, output_dir, target_size=TARGET_SIZE)


