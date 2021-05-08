import os
import sys
import argparse
import pickle

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tqdm
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_WARNINGS'] = '0'

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import skimage

import utils.helpers as helpers
from utils.predict_class import predict
from utils.eigenfiltering import eigenfiltering
from utils.preprocessing import preprocess_image, normalize_intensity, cut_intensity
from utils.bragg_filtering import bragg_filter_symmetry, isolate_bragg_peaks, ftp, bragg_filter_amorphous
from src.postprocessing import binarization_symmetry, binarization_localvariance, combine_symmetry_localvariance
from src.evaluate_segmentation import mdice_coef

# import src.utils.helpers as helpers
# from src.utils.predict_class import predict
# from src.utils.eigenfiltering import eigenfiltering
# from src.utils.preprocessing import preprocess_image

def run(parent_dir, model, output_dir_, TARGET_SIZE = 128, evaluate=True, verbose=True, run=False):
    """
    Segmentation with Eigenfilters
    :param parent_dir:
    :param model:
    :param output_dir:
    :return:
    """
    blurring = 16
    filtersize = 7

    output_dir = os.path.join(output_dir_, f'b{blurring}_fs{filtersize}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sum_all = 0
    sum_defective = 0

    all_value_p = []
    all_values_p = {0: [], 1: [], 2: []}

    all_value_f = []
    all_values_f = {0: [], 1: [], 2: []}


    for filecounter in tqdm.trange(len(os.listdir(parent_dir))):
        file = os.listdir(parent_dir)[filecounter]
        if file.endswith(".tif"):
            name = file[:-4]

            if run:
                all_defective = False

                # LOAD IMAGE AND PREPROCESS
                name = file[:-4]
                file = os.path.join(parent_dir, file)
                if verbose:
                    print("\nFile: " + file)
                original = helpers.get_image(file)
                plt.imsave(os.path.join(output_dir, name + "_original.png"), original, cmap='gray')
                plt.imsave(os.path.join(output_dir, name + "_01_original.svg"), original, cmap='gray')

                # # FOURIER SEGMENTATION
                if verbose:
                    print("Fourier Segmentation...")
                fourier_segmentation_amorphous = bragg_filter_amorphous(original)
                fourier_segmentation_symmetry =  bragg_filter_symmetry(original)*(-1)+1
                fourier_segmentation_total = np.maximum.reduce([fourier_segmentation_amorphous*2, fourier_segmentation_symmetry])
                np.save(os.path.join(output_dir, name + "fourier_segmentation.npy"), fourier_segmentation_total)
                plt.imshow(fourier_segmentation_total, vmin=0, vmax=2)
                plt.imsave(os.path.join(output_dir, name + "fourier_segmentation.svg"), fourier_segmentation_total, vmin=0, vmax=2)
                # plt.show()
                plt.close()

                # PREPROCESSING
                if verbose:
                    print("Preprocessing...")
                image = skimage.filters.gaussian(original, 2) - skimage.filters.gaussian(original, 3)
                image = np.interp(image, (image.min(), image.max()), (0, 254))
                plt.imsave(os.path.join(output_dir, name + "_02_preprocessed.svg"), image, cmap='gray')

                # CROP IN PATCHES
                STEP_SIZE = TARGET_SIZE
                patches = helpers.get_patches(image, target_size=TARGET_SIZE, step=STEP_SIZE)

                # PREDICT DEFECT ON PATCHES WITH CNN, THEN ACCUMULATE TO IMAGE
                if verbose:
                    print("\nPredicting {} patches...".format(len(patches)))
                classes = predict(patches, model)[:, 0].round().astype(int) # One-hot encoded
                n_patches = classes.shape[0]
                n_side = int(np.sqrt(n_patches))
                class_matrix = classes.T.reshape((n_side, n_side))
                score_map = helpers.map_scores(image, scores=classes, target_size=TARGET_SIZE, step_size=STEP_SIZE)
                fix, ax = plt.subplots(ncols=2)
                ax[0].imshow(original)
                ax[1].imshow(score_map)
                plt.imsave(os.path.join(output_dir, name + "_predicted.png"), score_map, cmap='viridis')
                plt.imsave(os.path.join(output_dir, name + "_03_predicted.svg"), score_map, cmap='viridis')
                # plt.show()
                plt.close()
                sum_all += n_patches
                sum_defective += len(classes[classes == 1.])

                # TRANSFORM TO GRAPH: PATCHES -> NODES. THEN FIND BEST NODE
                G_0, G_1, nx_pos = helpers.get_graph(class_matrix, connectivity=4)
                if verbose:
                    print("Defective: {}/{}".format(len(G_1), len(G_0)+len(G_1)))
                if len(G_0) == 0:
                    all_defective = True
                    # print("ATTENTION: All patches are defective. Aborting.")
                    best_node, value = helpers.get_best_node_from_Kneighbors(G_1, k=10, connectivity=4)
                    if value == 0.0:
                        print("Use 8-connectivity...")
                        G_0, G_1, nx_pos = helpers.get_graph(class_matrix, connectivity=8)
                        best_node, value = helpers.get_best_node_from_Kneighbors(G_1, k=10, connectivity=8)
                else:
                    best_node, value = helpers.get_best_node_from_Kneighbors(G_0, k=10, connectivity=4)
                    if value==0.0:
                        print("Use 8-connectivity...")
                        G_0, G_1, nx_pos = helpers.get_graph(class_matrix, connectivity=8)
                        best_node, value = helpers.get_best_node_from_Kneighbors(G_0, k=10, connectivity=8)
                graph_fig, graph_ax = helpers.draw_graphs([G_0, G_1], nx_pos=nx_pos)
                graph_fig.savefig(os.path.join(output_dir, name + "_04_graph.svg"))
                plt.close(graph_fig)

                # GET BEST PATCH
                best_patch = patches[best_node]
                np.save(os.path.join(output_dir, name + "bestpatch.npy"), best_patch)
                graph_fig, graph_ax = helpers.draw_graphs([G_0, G_1], nx_pos=nx_pos)
                graph_fig.savefig(os.path.join(output_dir, name + "_04_graph.svg"))
                plt.close(graph_fig)
                graph_fig_best, graph_ax_best = helpers.draw_graphs([G_0, G_1], nx_pos=nx_pos, extra_nodes=[best_node])
                # graph_fig_best.savefig(os.path.join(output_dir, name + "_05_graph_best.png"))
                graph_fig_best.savefig(os.path.join(output_dir, name + "_05_graph_best.svg"))
                # plt.show()
                plt.close(graph_fig_best)

                # # GET ANNOTADED IMAGE (WITH BEST PATCH)
                # image_annotated = add_bestpatch_to_img(image, patchsize=128, patch=best_node, color='gray')
                # score_map_annotated = add_bestpatch_to_img(score_map, patchsize=128, patch=best_node, color='viridis')

                # GET LOCAL VARIANCE OF BEST PATCH
                if verbose:
                    print("Local Variance...")
                localvariance_patch = helpers.localvariance_filter(image=best_patch)
                minlocalvariance = np.min(localvariance_patch)
                # FILTER FOR LOCAL VARIANCE
                localvariance = helpers.localvariance_filter(image=image)
                np.save(os.path.join(output_dir, name + "_08_localvariance.npy"), localvariance)
                plt.imsave(os.path.join(output_dir, name + "_08_localvariance.svg"), localvariance, cmap='inferno_r',
                           vmin=minlocalvariance - 300, vmax=minlocalvariance)

                # SEGMENTING: GET BEST PATCH AND USE IT AS AN EIGENFILTER ON THE IMAGE (will take 2-3min)
                if verbose:
                    print("Eigenfiltering...")
                startingtime = time.time()
                filtered = eigenfiltering(image_def=image, patch_good=best_patch, output_path=os.path.join(output_dir,
                                                                                                        'eigenfilters', name), blurring=blurring, filter_size=filtersize)
                maxfiltered = np.max(filtered)
                minval = np.min(filtered[filtered > 0])
                # plt.imshow(filtered, vmin=minval)
                # # plt.show()
                # plt.close()
                np.save(os.path.join(output_dir, name + "symmetry.npy"), filtered)
                plt.imsave(os.path.join(output_dir, name + "symmetry.svg"), filtered, cmap='viridis', vmin=minval)
                           # vmax=np.max((22, maxfiltered)))


            else:
                localvariance = np.load(os.path.join(output_dir, name + "_08_localvariance.npy"))
                best_patch = np.load(os.path.join(output_dir, name + "bestpatch.npy"))
                filtered = np.load(os.path.join(output_dir, name + "symmetry.npy"))
                fourier_segmentation_total = np.load(os.path.join(output_dir,
                                                                  name + "fourier_segmentation.npy"))

            binarized_symmetry = binarization_symmetry(filtered)
            binarized_localvariance = binarization_localvariance(localvariance, best_patch)
            binarized_both = combine_symmetry_localvariance(binarized_symmetry, binarized_localvariance)

            # plt.imshow(binarized_both)
            # # plt.show()
            # plt.close()
            np.save(os.path.join(output_dir, name[:-4] + '_binarized.npy'), binarized_both)

            if evaluate:
                border=12
                dir_label = "/home/nik/UZH/IBM/TEMDefectClassification/data/all_data/defective/labels"
                imagepath_label = os.path.join(dir_label, name + '.tif')
                image_label = (img_to_array(load_img(imagepath_label, color_mode="grayscale")) / 127).astype(
                    int).reshape([1024,
                                  1024])[border:-border, border:-border]

                # image_raw = original[b:-b, b:-b]
                image_predicted = binarized_both[border:-border, border:-border]
                image_fourier = fourier_segmentation_total[border:-border, border:-border]
                image_predicted[image_predicted == 1.05] = 2
                image_predicted = image_predicted.astype(int)
                image_fourier = image_fourier.astype(int)

                labels = np.unique(image_label)
                value_p, values_p = mdice_coef(image_label, image_predicted)
                value_f, values_f = mdice_coef(image_label, image_fourier)

                print(value_p)
                print(values_p)
                print(value_f)
                print(values_f)

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

    if evaluate:
        print(all_value_p)
        print(f"mean predicted: {np.mean(all_value_p)}")
        for key, value in all_values_p.items():
            print(f"{key}: {np.mean(value)}")

        print(all_value_f)
        print(f"mean predicted: {np.mean(all_value_f)}")
        for key, value in all_values_f.items():
            print(f"{key}: {np.mean(value)}")

        np.save(os.path.join(output_dir, "all_value_f.npy"), all_value_f)
        np.save(os.path.join(output_dir, "all_value_p.npy"), all_value_p)

        with open(os.path.join(output_dir, 'all_values_f.json'), 'wb') as fp:
            pickle.dump(all_values_f, fp)

        with open(os.path.join(output_dir, 'all_values_p.json'), 'wb') as fp:
            pickle.dump(all_values_p, fp)

if __name__ == '__main__':

    image_dir = "../data/all_data/defective/images"
    output_dir = "../output/all_data/fold0"             # Change fold as desired
    model_path = "../models/all_data2/fold0_20210325-005303/model.h5"
    assert os.path.exists(model_path)
    model = load_model(model_path)
    run(image_dir, model, output_dir)

    print("Done")
    exit()




    #
    #
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("-image_dir", "--image_dir", type=str, default="cubic/defective/images",
    # #                     help="Name of image directory")
    # # parser.add_argument("-image_dir", "--image_dir", type=str, default="../data/new/Images from PEW1",
    # #                     help="Name of image directory")
    #
    #
    # # parser.add_argument("-image_dir", "--image_dir", type=str, default="../data/all_data/defective/images",
    # #                     help="Name of image directory")
    # parser.add_argument("-image_dir", "--image_dir", type=str, default="../data/test99",
    #                     help="Name of image directory")
    #
    # # parser.add_argument("-image_dir", "--image_dir", type=str, default="12",
    # #                     help="Name of image directory")
    # # parser.add_argument("-image_dir", "--image_dir", type=str, default="../data/test_bragg",
    # #                     help="Name of image directory")
    # # parser.add_argument("-image_dir", "--image_dir", type=str, default="test2",
    # #                     help="Name of image directory")
    # # parser.add_argument("-model", "--model", type=str, default="../models/20191208-014141.h5",
    # #                     help="Name of model file.")
    #
    # # parser.add_argument("-model", "--model", type=str,
    # #                     default="../models/all_data/n_train_16/fold9_20210311-134051/model.h5")
    # # parser.add_argument("-model", "--model", type=str,
    # #                     default="../models/all_data/n_train_16/newtraining/fold2_20210317-233638/model.h5")
    # parser.add_argument("-model", "--model", type=str,
    #                     default="../models/all_data2/fold0_20210325-005303/model.h5") # BEST RECALL
    # # parser.add_argument("-model", "--model", type=str,
    # #                     default="../models/all_data2/fold7_20210325-165411/model.h5")
    #
    # # parser.add_argument("-output_dir", "--output_dir", type=str, default="eigenfilter_segmentation_forpost",
    # #                     help="Name of model file.")
    # # parser.add_argument("-output_dir", "--output_dir", type=str, default="new_Images_from_PEW1",
    # #                     help="Name of model file.")
    # # parser.add_argument("-output_dir", "--output_dir", type=str,
    # #                     default="../output/all_data_verynewmodel_ntrain22_differentgraphlogic/fold0/",
    # #                     help="Name of output dir")
    # parser.add_argument("-output_dir", "--output_dir", type=str,
    #                     default="../output/all_data_verynewmodel_ntrain22_differentgraphlogic/test99/",
    #                     help="Name of output dir")
    # args = parser.parse_args()
    #
    # image_dir = args.image_dir #os.path.join('data', args.image_dir)
    # assert os.path.exists(image_dir)
    #
    # output_dir = args.output_dir#os.path.join('output', args.output_dir)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # model_path = args.model#os.path.join('models', args.model)
    # assert os.path.exists(model_path)
    # model = load_model(model_path)
    #
    # print(image_dir)
    # run(image_dir, model, output_dir)
    # print("Done")








