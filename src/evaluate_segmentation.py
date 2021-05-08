import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.image import save_img
# from keras.preprocessing.image import

from keras import backend as K
from matplotlib.path import Path as mPath
import matplotlib.patches as mpatches
plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['legend.title_fontsize'] = 'small'
# plt.rcParams['legend.fontsize'] = 'small'


import numpy as np
import matplotlib.pyplot as plt

def mdice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    labels = np.unique(y_true_f)

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef3(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (union + smooth)

# def iou_coef(y_true, y_pred, smooth=1):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(np.abs(y_true_f * y_pred_f))
#     union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
#     iou = np.mean((intersection + smooth) / (union + smooth))
#     return iou

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index])
    return dice/numLabels # taking average

def show_both(img_label, img_fourier, img_predicted, name, dicevalue_f, dicevalue_p):
    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(img_label, vmax=2)
    ax[0].set_title("Ground Truth")

    ax[1].imshow(img_fourier, vmax=2)
    ax[1].set_title(f"Fourier: {np.round(dicevalue_f, 4)}")

    ax[2].imshow(img_predicted, vmax=2)
    ax[2].set_title(f"Predicted: {np.round(dicevalue_p, 4)}")

    for axx in ax:
        axx.set_xticks([])
        axx.set_yticks([])

    plt.suptitle(f"{name}")
    plt.show()

def show_all(img_raw, img_label, img_fourier, img_predicted, name, dice_f, dice_p, output_dir):

    cm_label = cm.get_cmap('gist_earth_r')
    cm_label.set_under('white', alpha=0)

    if False:
        fig, ax = plt.subplots()
        ax.imshow(img_raw, cmap='gray')
        # ax.imshow(img_label, vmin=0.01, vmax=6, alpha=0.6, cmap=cm_label)
        # ax[1].set_under('white', alpha=0)
        # ax[1].set_title("Ground Truth")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(output_dir, name+"_original.png"), dpi=300)
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(img_raw, cmap='gray')
        ax.imshow(img_label, vmin=0.01, vmax=6, alpha=0.6, cmap=cm_label)
        # ax[1].set_under('white', alpha=0)
        # ax[1].set_title("Ground Truth")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(output_dir, name+"_label_true.png"), dpi=300)
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(img_raw, cmap='gray')
        ax.imshow(img_fourier, vmin=0.01, vmax=6, alpha=0.6, cmap=cm_label)
        # ax[1].set_under('white', alpha=0)
        # ax[1].set_title("Ground Truth")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(output_dir, name+"_label_bragg.png"), dpi=300)
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(img_raw, cmap='gray')
        ax.imshow(img_predicted, vmin=0.01, vmax=6, alpha=0.6, cmap=cm_label)
        # ax[1].set_under('white', alpha=0)
        # ax[1].set_title("Ground Truth")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(output_dir, name+"_label_ours.png"), dpi=300)
        plt.show()

    int_dict = {'B': 0, 'S': 1, 'A': 2}
    label_dict = {'B': 'primary symmetry', 'S': 'secondary symmetry', 'A': 'blurred'}

    fig, ax = plt.subplots(ncols=4)

    ax[0].imshow(img_raw, cmap='gray')

    ax[1].imshow(img_raw, cmap='gray')
    ax[1].imshow(img_label, vmin=0.01, vmax=6, alpha=0.6, cmap=cm_label)
    # ax[1].set_under('white', alpha=0)

    ax[2].imshow(img_raw, cmap='gray')
    ax[2].imshow(img_fourier, vmin=0.01, vmax=6, alpha=0.6, cmap=cm_label)
    # ax[2].set_title(f"Fourier: {np.round(dice_f, 4)}")


    ax[3].imshow(img_raw, cmap='gray')
    im = ax[3].imshow(img_predicted, vmin=0.01, vmax=6, alpha=0.6, cmap=cm_label)
    # ax[3].set_title(f"Predicted: {np.round(dice_p, 4)}")

    for axx in ax:
        axx.set_xticks([])
        axx.set_yticks([])

    size = fig.get_size_inches() * fig.dpi

    print(size)
    ax[0].set_title("Raw", fontsize = 'small')
    ax[1].set_title("Ground Truth", fontsize = 'small')
    ax[2].set_title(f"FFT-BF", fontsize = 'small')
    ax[3].set_title(f"CNN+EF", fontsize = 'small')

    # ax[2].text(f"DICE: {np.round(dice_f, 2):.2f}")

    ax[2].text(0.04, 0.85, f"DICE: {np.round(dice_f, 2):.2f}",
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax[2].transAxes,
        color='w', fontsize=6, weight='bold')

    ax[3].text(0.04, 0.85, f"DICE: {np.round(dice_p, 2):.2f}",
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax[3].transAxes,
        color='w', fontsize=6, weight='bold')

    # ax[2].set_xlabel(f"DICE: {np.round(dice_f, 2):.2f}", y=-0.25, fontsize = 'smaller')
    # ax[3].set_xlabel(f"DICE: {np.round(dice_p, 2):.2f}", y=-0.25, fontsize = 'smaller')

    values = np.unique([img_label.ravel(), img_fourier.ravel(), img_predicted.ravel()])
    print(values)
    print(int_dict)
    print(label_dict)
    # get the colors of the values, according to the
    # colormap used by imshow
    int_dict_reversed = {v: k for k, v in int_dict.items()}
    patches = [mpatches.Patch(color=im.cmap(value),
                              label=label_dict[int_dict_reversed[value]]) for
               value in
               values]
    # put those patched as legend-handles into the legend
    # ax[1].legend(handles=patches, bbox_to_anchor=(0.73, -0.02), loc=0, borderaxespad=0.)



    colors = [cm_label(0), cm_label(1/6), cm_label(2/6)]
    texts = ['Non-Defective', 'Symmetry Defect', 'Blurred Area']
    patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color='black', markerfacecolor=colors[i],
                        linewidth=0.1,
                        label="{:s}".format(texts[i]))[0] for i in range(len(texts))]
    fig.legend(handles=patches, frameon=False,
               loc='center right', numpoints=1, borderaxespad=0.1)

    plt.subplots_adjust(right=0.72)

    plt.suptitle(f"{name}")
    # plt.tight_layout()

    plt.savefig(os.path.join(output_dir, name+ "_all.png"), dpi=600)
    plt.savefig(os.path.join(output_dir, name+ "_all.svg"), dpi=600)
    plt.show(dpi=400)




# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
#   union = K.sum(y_true)+K.sum(y_pred)-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou
#
# def dice_coef2(y_true, y_pred, smooth=1):
#   intersection = K.sum(y_true * y_pred)
#   union = K.sum(y_true) + K.sum(y_pred)
#   dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
#   return dice

def dice_coef(img, img2):
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:

        lenIntersection = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (np.array_equal(img[i][j], img2[i][j])):
                    lenIntersection += 1

        lenimg = img.shape[0] * img.shape[1]
        lenimg2 = img2.shape[0] * img2.shape[1]
        value = (2. * lenIntersection / (lenimg + lenimg2))
    return value


def mdice_coef(img, img2):
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:

        values = []
        for label in np.unique(img):
            lenA = 0
            lenB = 0
            lenIntersection = 0
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] == label:
                        lenA += 1
                    if img2[i][j] == label:
                        lenB += 1
                    if (img[i][j] == img2[i][j] == label):
                        lenIntersection += 1
            value = (2. * lenIntersection / (lenA + lenB))
            values.append(value)
        meanvalue = np.mean(values)
    return meanvalue, values

def noise_analysis(output_dir_noise):

    # fig, ax = plt.subplots()
    # fig, ax = plt.subplots(ncols=2)

    all_a = []
    all_f_0 = []
    all_f_1 = []
    all_f_2 = []
    all_f_m = []
    all_p_0 = []
    all_p_1 = []
    all_p_2 = []
    all_p_m = []

    for a in sorted([float(a) for a in os.listdir(output_dir_noise)]):
        name = str(a)
        # a = float(name)
        all_a.append(a)

        dir = os.path.join(output_dir_noise, name, 'b16_fs7')

        all_value_f = np.load(os.path.join(dir, 'all_value_f.npy'))
        all_value_p = np.load(os.path.join(dir, 'all_value_p.npy'))

        with open(os.path.join(dir, 'all_values_f.json'), 'rb') as fp:
            all_values_f = pickle.load(fp)
        with open(os.path.join(dir, 'all_values_p.json'), 'rb') as fp:
            all_values_p = pickle.load(fp)

        # print(all_value_f)
        # print(all_value_p)
        # print(all_values_f)
        # print(all_values_p)
        # exit()
        #

        dice_f_0_mean = np.mean(all_values_f[0])
        dice_f_0_std = np.std(all_values_f[0])
        dice_p_0_mean = np.mean(all_values_p[0])
        dice_p_0_std = np.std(all_values_p[0])

        dice_f_1_mean = np.mean(all_values_f[1])
        dice_f_1_std = np.std(all_values_f[1])
        dice_p_1_mean = np.mean(all_values_p[1])
        dice_p_1_std = np.std(all_values_p[1])

        dice_f_2_mean = np.mean(all_values_f[2])
        dice_f_2_std = np.std(all_values_f[2])
        dice_p_2_mean = np.mean(all_values_p[2])
        dice_p_2_std = np.std(all_values_p[2])

        dice_f_mean = np.mean(all_value_f)
        dice_f_std = np.std(all_value_f)
        dice_p_mean = np.mean(all_value_p)
        dice_p_std = np.std(all_value_p)

        all_f_0.append(dice_f_0_mean)
        all_f_1.append(dice_f_1_mean)
        all_f_2.append(dice_f_2_mean)
        all_f_m.append(dice_f_mean)
        all_p_0.append(dice_p_0_mean)
        all_p_1.append(dice_p_1_mean)
        all_p_2.append(dice_p_2_mean)
        all_p_m.append(dice_p_mean)


        # ax[0].errorbar(a, dice_f_0_mean, dice_f_0_std, c='k', capsize=2, alpha=0.3)
        # ax[0].errorbar(a, dice_f_1_mean, dice_f_1_std, c='y', capsize=2, alpha=0.3)
        # ax[0].errorbar(a, dice_f_2_mean, dice_f_2_std, c='r', capsize=2, alpha=0.3)
        # ax[1].errorbar(a, dice_p_0_mean, dice_p_0_std, c='k', capsize=2, alpha=0.3)
        # ax[1].errorbar(a, dice_p_1_mean, dice_p_1_std, c='y', capsize=2, alpha=0.3)
        # ax[1].errorbar(a, dice_p_2_mean, dice_p_2_std, c='r', capsize=2, alpha=0.3)
        #
        # ax[0].scatter(a, dice_f_0_mean, c='k')
        # ax[0].scatter(a, dice_f_1_mean, c='y')
        # ax[0].scatter(a, dice_f_2_mean, c='r')
        # ax[1].scatter(a, dice_p_0_mean, c='k')
        # ax[1].scatter(a, dice_p_1_mean, c='y')
        # ax[1].scatter(a, dice_p_2_mean, c='r')

        # ax.errorbar(a, dice_f_mean, dice_f_std, c='firebrick', capsize=2, alpha=0.3)
        # ax.errorbar(a, dice_p_mean, dice_p_std, c='indigo', capsize=2, alpha=0.3)
        # # ax.plot(a, dice_f_mean, c='firebrick')
        # # ax.plot(a, dice_p_mean, c='indigo')
        # ax.scatter(a, dice_f_mean, c='firebrick', label='FFT-BF')
        # ax.scatter(a, dice_p_mean, c='indigo', label='CNN+EF')

    # ax[0].set_ylim([-0.19, 1.19])
    # ax[1].set_ylim([-0.19, 1.19])
    #
    # ax[0].set_xscale('log')
    # ax[0].invert_xaxis()
    # ax[1].set_xscale('log')
    # ax[1].invert_xaxis()
    # ax.set_xscale('log')
    # ax.invert_xaxis()

    # ax.set_ylabel("DICE")
    # ax.set_xlabel(r"Noise Parameter $\alpha$")

    # handles, labels = ax[0].get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    #
    # # plt.legend()
    # plt.show()

    # fig, ax = plt.subplots()
    # cmap = cm.get_cmap('Spectral')
    # cm_label = cm.get_cmap('gist_earth_r')
    # cm_label.set_under('white', alpha=0)

    dir = "/home/nik/UZH/IBM/TEMDefectClassification/output/all_data_verynewmodel_ntrain22_differentgraphlogic" \
              "/fold0" \
              "/b16_fs7"
    all_value_f = np.load(os.path.join(dir, 'all_value_f.npy'))
    all_value_p = np.load(os.path.join(dir, 'all_value_p.npy'))

    with open(os.path.join(dir, 'all_values_f.json'), 'rb') as fp:
        all_values_f = pickle.load(fp)
    with open(os.path.join(dir, 'all_values_p.json'), 'rb') as fp:
        all_values_p = pickle.load(fp)

    dice_f_0_mean = np.mean(all_values_f[0])
    dice_f_0_std = np.std(all_values_f[0])
    dice_p_0_mean = np.mean(all_values_p[0])
    dice_p_0_std = np.std(all_values_p[0])

    dice_f_1_mean = np.mean(all_values_f[1])
    dice_f_1_std = np.std(all_values_f[1])
    dice_p_1_mean = np.mean(all_values_p[1])
    dice_p_1_std = np.std(all_values_p[1])

    dice_f_2_mean = np.mean(all_values_f[2])
    dice_f_2_std = np.std(all_values_f[2])
    dice_p_2_mean = np.mean(all_values_p[2])
    dice_p_2_std = np.std(all_values_p[2])

    dice_f_mean = np.mean(all_value_f)
    dice_f_std = np.std(all_value_f)
    dice_p_mean = np.mean(all_value_p)
    dice_p_std = np.std(all_value_p)

    print("f")
    print(dice_f_0_mean, dice_f_0_std)
    print(dice_f_1_mean, dice_f_1_std)
    print(dice_f_2_mean, dice_f_2_std)
    print(dice_f_mean, dice_f_std)

    print("p")
    print(dice_p_0_mean, dice_p_0_std)
    print(dice_p_1_mean, dice_p_1_std)
    print(dice_p_2_mean, dice_p_2_std)
    print(dice_p_mean, dice_p_std)
    print(all_value_p)

    exit()


    red = 'indianred'#cm_label(1 / 6)
    yellow = 'gold'#cm_label(2 / 6)

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)

    print(all_a)
    ax[0].scatter(all_a, all_f_0, c='w', label='Non-Defective', alpha=1, marker='o', edgecolors='k', s=18)
    ax[0].scatter(all_a, all_f_1, c=red, label='Symmetry Defect', alpha=1, marker='o', edgecolors='k', s=18)
    ax[0].scatter(all_a, all_f_2, c=yellow, label='Blurred Area', alpha=1, marker='o', edgecolors='k', s=18)
    ax[0].scatter(all_a, all_f_m, c='k', label='Total', alpha=1, marker='o', edgecolors='k')

    ax[1].scatter(all_a, all_p_0, c='w', label='non-defective', alpha=1, marker='o', edgecolors='k', s=18)
    ax[1].scatter(all_a, all_p_1, c=red, label='symmetry defect', alpha=1, marker='o', edgecolors='k', s=18)
    ax[1].scatter(all_a, all_p_2, c=yellow, label='blurry area', alpha=1, marker='o', edgecolors='k', s=18)
    ax[1].scatter(all_a, all_p_m, c='k', label='total', alpha=1, marker='o', edgecolors='k')


    ax[0].scatter(10, dice_f_0_mean, c='w', alpha=1, marker='o', edgecolors='k', s=18)
    ax[0].scatter(10, dice_f_1_mean, c=red, alpha=1, marker='o', edgecolors='k', s=18)
    ax[0].scatter(10, dice_f_2_mean, c=yellow, alpha=1, marker='o', edgecolors='k', s=18)
    ax[0].scatter(10, dice_f_mean, c='k', alpha=1, marker='o', edgecolors='k')
    ax[0].axvline(x=3, color='k', linestyle='--', alpha=0.3)

    ax[1].scatter(10, dice_p_0_mean, c='w', alpha=1, marker='o', edgecolors='k', s=18)
    ax[1].scatter(10, dice_p_1_mean, c=red, alpha=1, marker='o', edgecolors='k', s=18)
    ax[1].scatter(10, dice_p_2_mean, c=yellow, alpha=1, marker='o', edgecolors='k', s=18)
    ax[1].scatter(10, dice_p_mean, c='k', alpha=1, marker='o', edgecolors='k')
    ax[1].axvline(x=3, color='k', linestyle='--', alpha=0.3)





    ax[0].set_xscale('log')

    ax[0].set_xticks([1, 0.1, 0.01, 0.001])
    ax[1].set_xticks([1, 0.1, 0.01, 0.001])

    ax[0].set_xticks([0.001, 0.01, 0.1, 1, 10])
    ax[1].set_xticks([0.001, 0.01, 0.1, 1, 10])

    # ax[0].set_xticklabels([0.001, 0.01, 0.1, 1, 'raw'])
    # ax[1].set_xticklabels([0.001, 0.01, 0.1, 1, 'raw'])

    # ax[0].ticklabel_format(style='sci', axis='x')

    # fig.show()
    # fig.canvas.draw()
    #
    # labels = [tick.get_text() for tick in ax[0].get_xticklabels()]
    # print(labels)
    # ax[0].set_xticklabels(labels[:-1])

    # fig.show()
    # fig.canvas.draw()
    # ax[0].set_xticklabels([r"$\\mathdefault{10^{-3}}$", r"$\\mathdefault{10^{-2}}$", r"$\\mathdefault{10^{-1}}$",
    #                        r"$\\mathdefault{10^{0}}$", 'raw'])
    ax[1].set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", 'raw'])
    # ax[1].set_xticklabels([r"$\\mathdefault{10^{-3}}$", r"$\\mathdefault{10^{-2}}$", r"$\\mathdefault{10^{-1}}$",
    #                        r"$\\mathdefault{10^{0}}$", 'raw'])

    # ax[0].invert_xaxis()
    ax[0].set_ylim([-0.09, 1.09])
    ax[0].set_xlim([10*2, 0.001/2])


    # ax[1].set_xscale('log')
    # ax[1].set_xlim(0.001 / 2, 1 * 2)
    # ax[1].invert_xaxis()
    # ax[1].set_ylim([-0.19, 1.19])

    ax[0].legend(loc='upper left')
    ax[0].set_ylabel("DICE Score")
    ax[0].set_xlabel(r"Noise Parameter $\alpha$")
    ax[1].set_xlabel(r"Noise Parameter $\alpha$")

    ax[0].set_title("FFT-BF")
    ax[1].set_title("CNN+EF")


    plt.savefig(os.path.join(output_dir_noise, "dice_noise.svg"))
    plt.savefig(os.path.join(output_dir_noise, "dice_noise.png"), dpi=300)
    plt.show()
        #
        #
        # exit()



if __name__ == '__main__':
    output_dir_noise = '/home/nik/UZH/IBM/TEMDefectClassification/output/all_data/noise_analysis'
    noise_analysis(output_dir_noise)


    exit()

    imagepath_label = "/home/nik/UZH/IBM/TEMDefectClassification/data/all_data/defective/labels" \
                      "/JEOL_ADF1_10M_SOU76posB_Tunnel5Seed1DefectMeet1.tif"
    imagepath_predicted = '/home/nik/UZH/IBM/TEMDefectClassification/output/eigenfilter_segmentation_12/b16_fs7' \
                      '/JEOL_BF_10M_SOU76posB_Tunnel5Seed1DefectMeet1symmetry_binarized.npy'
    dir_raw = '/home/nik/UZH/IBM/TEMDefectClassification/data/all_data/defective/images'
    dir_label = "/home/nik/UZH/IBM/TEMDefectClassification/data/all_data/defective/labels"
    dir_predicted = '/home/nik/UZH/IBM/TEMDefectClassification/output/all_data/b16_fs7'
    dir_predicted = '/home/nik/UZH/IBM/TEMDefectClassification/output/all_data_verynewmodel/fold2/b16_fs7'
    dir_predicted = '/home/nik/UZH/IBM/TEMDefectClassification/output/all_data_verynewmodel_ntrain22/fold0/b16_fs7'
    dir_predicted = "/home/nik/UZH/IBM/TEMDefectClassification/output/all_data_verynewmodel_ntrain22_differentgraphlogic/fold0/b16_fs7"
    # dir_fourier = '/home/nik/UZH/IBM/TEMDefectClassification/output/all_data/b16_fs7'
    dir_fourier = "/home/nik/UZH/IBM/TEMDefectClassification/output" \
                  "/all_data_verynewmodel_ntrain22_differentgraphlogic/fold0/b16_fs7"


    dir_output = '/home/nik/UZH/IBM/TEMDefectClassification/output/all_data_verynewmodel/fold2/b16_fs7/final'
    dir_output = '/home/nik/UZH/IBM/TEMDefectClassification/output/all_data_verynewmodel_ntrain22/fold0/b16_fs7'
    dir_output = "/home/nik/UZH/IBM/TEMDefectClassification/output/all_data_verynewmodel_ntrain22_differentgraphlogic/fold0/b16_fs7"
    dir_output = "/home/nik/UZH/IBM/TEMDefectClassification/output/_PUBLICATION/Figure 4"

    if True:
        all_value_p = []
        all_values_p = {0: [], 1: [], 2: []}

        all_value_f = []
        all_values_f = {0: [], 1: [], 2: []}

        i = 0
        for path in os.listdir(dir_label):
            i += 1
            name = path.split(sep='/')[-1][:-4]
            print(name)
            imagepath_raw = os.path.join(dir_raw, name + '.tif')
            imagepath_label = os.path.join(dir_label, name + '.tif')
            imagepath_predicted = os.path.join(dir_predicted, name + 'symmetry_binarized.npy')
            imagepath_fourier = os.path.join(dir_fourier, name + 'fourier_segmentation.npy')

            b = 12
            #
            # image_raw = (img_to_array(load_img(imagepath_raw, color_mode="grayscale"))).reshape([1024, 1024])[b:-b, b:-b]
            #
            # plt.imshow(image_raw)
            # plt.show()
            # exit()

            image_label = (img_to_array(load_img(imagepath_label, color_mode="grayscale"))/127).astype(int).reshape([1024,
                                                                                                                     1024])[b:-b, b:-b]

            image_raw = (img_to_array(load_img(imagepath_raw, color_mode="grayscale"))).reshape([1024, 1024])[b:-b, b:-b]

            # image_label = np.load(imagepath_label)
            try:
                image_predicted = np.load(imagepath_predicted).reshape([1024, 1024])[b:-b, b:-b]
                image_fourier = np.load(imagepath_fourier).reshape([1024, 1024])[b:-b, b:-b]
            except FileNotFoundError:
                print(f"Have not found {name}")
                continue
            image_predicted[image_predicted==1.05] = 2
            image_predicted = image_predicted.astype(int)
            image_fourier = image_fourier.astype(int)

            labels = np.unique(image_label)
            # print(np.unique(image_predicted))

            # print(dice_coef(image_label, image_predicted))
            value_p, values_p = mdice_coef(image_label, image_predicted)
            value_f, values_f = mdice_coef(image_label, image_fourier)

            print(value_p)
            print(values_p)

            print(value_f)
            print(values_f)
            # show_both(image_label, image_fourier, image_predicted, name, value_f, value_p)
            if True:
                show_all(image_raw, image_label, image_fourier, image_predicted, name, value_f, value_p, dir_output)
            all_value_p.append(value_p)
            all_value_f.append(value_f)


            all_values_p[labels[0]].append(values_p[0])
            all_values_f[labels[0]].append(values_f[0])

            try:
                all_values_p[labels[1]].append(values_p[1])
            except IndexError:
                pass
            try:
                all_values_p[labels[2]].append(values_p[2])
            except IndexError:
                pass

            try:
                all_values_f[labels[1]].append(values_f[1])
            except IndexError:
                pass
            try:
                all_values_f[labels[2]].append(values_f[2])
            except IndexError:
                pass

            print("")
            # if i==2:
            #     break

        print(all_value_p)
        print(f"mean predicted: {np.mean(all_value_p)}")
        for key, value in all_values_p.items():
            print(f"{key}: {np.mean(value)}")

        print(all_value_f)
        print(f"mean predicted: {np.mean(all_value_f)}")
        for key, value in all_values_f.items():
            print(f"{key}: {np.mean(value)}")




        np.save(os.path.join(dir_output, "all_value_f.npy"), all_value_f)
        np.save(os.path.join(dir_output, "all_value_p.npy"), all_value_p)

    all_value_f = np.load(os.path.join(dir_output, "all_value_f.npy"))
    all_value_p = np.load(os.path.join(dir_output, "all_value_p.npy"))

    fig, ax = plt.subplots()
    # ax.hist([all_value_f, all_value_p], bins=20, range = [0,1], color ='firebrick', lw=0, label='FFT-BF', alpha=0.5, \
    #                                                                                                  stacked=True, density = True)
    # ax.hist(all_value_p, bins=20, range = [0,1], color ='indigo', lw=0, label='CNN+EF', alpha=0.5, stacked=True, density = True)
    bins = (np.arange(7))/6
    n = ax.hist([ all_value_f, all_value_p], bins=bins,  range = [0,1], color =[ 'firebrick', 'indigo'], lw=0,
            label=['FFT-BF', 'CNN+EF'],
            alpha=0.7, \
                                                                                                     stacked=False)
    # ax.set_xticks(np.arange(9)/8)

    print(n)
    print(n[1])
    # print(bins)
    # exit()
    # ax.hist([ all_value_p, all_value_f], bins=14,  range = [0,1], color =['indigo', 'firebrick'], lw=0,
    #         label=['CNN+EF', 'FFT-BF'],
    #         alpha=0.7, \
    #                                                                                                  stacked=True)
    # ax.set_xticks(n[1]+1/12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    ax.set_ylabel("Occurence")
    ax.set_xlabel("DICE Score")

    plt.legend()

    plt.savefig(os.path.join(dir_output, "histogram.png"), dpi=300)
    plt.savefig(os.path.join(dir_output, "histogram.svg"))

    plt.show()

    # print(dice_coef3(image_label, image_predicted))
        # print(dice_coef_multilabel(image_label, image_predicted, len(np.unique(image_label))))
        # print(iou_coef(image_label, image_predicted))

    # print(dice_coef(image_label, image_label))
    # print(dice_coef3(image_predicted, image_predicted))
    # print(iou_coef(image_predicted, image_predicted))
    # print(dice_coef2(image_label, image_predicted))
    # print(iou_coef(image_label, image_predicted))
    # print(dice)
    exit()
    pass