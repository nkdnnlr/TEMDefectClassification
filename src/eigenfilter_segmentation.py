import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tqdm
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_WARNINGS'] = '0'

import matplotlib.pyplot as plt
from keras.models import load_model

import src.utils.helpers as helpers
from src.utils.predict_class import predict
from src.utils.eigenfiltering import eigenfiltering
from src.utils.preprocessing import preprocess_image


def run(parent_dir, model, output_dir):
    """
    Segmentation with Eigenfilters
    :param parent_dir:
    :param model:
    :param output_dir:
    :return:
    """
    sum_all = 0
    sum_defective = 0

    for filecounter in tqdm.trange(len(os.listdir(parent_dir))):
        file = os.listdir(parent_dir)[filecounter]
        if file.endswith(".tif"):
            all_defective = False

            # LOAD IMAGE AND PREPROCESS
            name = file[:-4]
            file = os.path.join(parent_dir, file)
            print("\nFile: " + file)
            original = helpers.get_image(file)

            image = preprocess_image(original,
                             lowpass_filter="gaussian", lowpass_kernel_size=5,
                             highpass_filter="laplacian", highpass_kernel_size=19,
                             rescale=False)

            # # DRAW PATCHED IMAGE
            # fig, ax = draw_image_patched(image, patchsize=TARGET_SIZE)
            # plt.savefig(os.path.join(output_dir, name + "_preprocessed_patched.png"), dpi=300, bbox_inches='tight')
            # plt.close(fig)

            # CROP IN PATCHES
            startingtime = time.time()
            STEP_SIZE = TARGET_SIZE
            patches = helpers.get_patches(image, target_size=TARGET_SIZE, step=STEP_SIZE)

            # PREDICT DEFECT ON PATCHES WITH CNN, THEN ACCUMULATE TO IMAGE
            print("\nPredicting {} patches...".format(len(patches)))
            classes = predict(patches, model)[:, 0].round().astype(int) # One-hot encoded
            print("PREDICTIONTIME: ", time.time()-startingtime)
            # continue
            n_patches = classes.shape[0]
            n_side = int(np.sqrt(n_patches))
            class_matrix = classes.T.reshape((n_side, n_side))
            score_map = helpers.map_scores(image, scores=classes, target_size=TARGET_SIZE, step_size=STEP_SIZE)
            # plt.imsave(os.path.join(output_dir, name + "_predicted.png"), score_map, cmap='viridis')
            sum_all += n_patches
            sum_defective += len(classes[classes == 1.])

            # TRANSFORM TO GRAPH: PATCHES -> NODES. THEN FIND BEST NODE
            G_0, G_1, nx_pos = helpers.get_graph(class_matrix, connectivity=4)
            print("Defective: {}/{}".format(len(G_1), len(G_0)+len(G_1)))
            if len(G_0) == 0:
                all_defective = True
                print("ATTENTION: All patches are defective. Aborting.")
                continue


            # best_node = get_most_central_node(G_0)
            best_node = helpers.get_best_node_from_Kneighbors(G_0, k=10)

            # GET BEST PATCH
            best_patch = patches[best_node]

            # # GET ANNOTADED IMAGE (WITH BEST PATCH)
            # image_annotated = add_bestpatch_to_img(image, patchsize=128, patch=best_node, color='gray')
            # score_map_annotated = add_bestpatch_to_img(score_map, patchsize=128, patch=best_node, color='viridis')

            # GET LOCAL VARIANCE OF BEST PATCH
            localvariance_patch = helpers.localvariance_filter(image=best_patch)
            minlocalvariance = np.min(localvariance_patch)

            # FILTER FOR LOCAL VARIANCE
            localvariance = helpers.localvariance_filter(image=image)
            print("hi?")
            # SEGMENTING: GET BEST PATCH AND USE IT AS AN EIGENFILTER ON THE IMAGE
            startingtime = time.time()
            filtered = eigenfiltering(image_def=image, patch_good=best_patch, output_path=os.path.join(output_dir,
                                                                                                       'eigenfilters', name))
            print("FILTERINGTIME: ", time.time() - startingtime)
            continue
            print("hi!")
            maxfiltered = np.max(filtered)

            # SAVE PLOTS
            plt.imsave(os.path.join(output_dir, name + "_01_original.svg"), original, cmap='gray')
            plt.imsave(os.path.join(output_dir, name + "_02_preprocessed.svg"), image, cmap='gray')
            plt.imsave(os.path.join(output_dir, name + "_03_predicted.svg"), score_map, cmap='viridis')
            graph_fig, graph_ax = helpers.draw_graphs([G_0, G_1], nx_pos=nx_pos)
            graph_fig.savefig(os.path.join(output_dir, name + "_04_graph.svg"))
            plt.close(graph_fig)
            graph_fig_best, graph_ax_best = helpers.draw_graphs([G_0, G_1], nx_pos=nx_pos, extra_nodes=[best_node])
            # graph_fig_best.savefig(os.path.join(output_dir, name + "_05_graph_best.png"))
            graph_fig_best.savefig(os.path.join(output_dir, name + "_05_graph_best.svg"))
            plt.close(graph_fig_best)
            plt.imsave(os.path.join(output_dir, name + "_06_best_patch.svg"), best_patch, cmap='gray')
            plt.imsave(os.path.join(output_dir, name + "_07_segmented.svg"), filtered, cmap='viridis', vmin=7,
                       vmax=np.max((22, maxfiltered)))
            plt.imsave(os.path.join(output_dir, name + "_08_variance.svg"), localvariance, cmap='inferno_r',
                       vmin=minlocalvariance-300, vmax=minlocalvariance)

            helpers.make_publication_subplots(original, score_map, filtered, localvariance, 128,
                                      best_node, maxfiltered, minlocalvariance,
                                      output_path=os.path.join(output_dir, name + "_09_all.png"))
            plt.close()

    print("sum_all: ", sum_all)
    print("sum_defective: ", sum_defective)
    print("fraction_defective: ", 1.-sum_defective/sum_all)


if __name__ == '__main__':
    TARGET_SIZE = 128

    defective_dir = "../data/cubic/defective/images/"
    nondefective_dir = "../data/cubic/non_defective/images"

    output_dir = "../output/eigenfilter_segmentation_svg/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # #Plot all raw images
    # plot_all_raw_images(nondefective_dir, defective_dir, output_dir=output_dir)

    parent_dir_models = "../models/finetuned/"
    path_bestmodel = "../models/finetuned/20191208-014141/model.h5"
    assert os.path.exists(path_bestmodel)
    model = load_model(path_bestmodel)

    run(defective_dir, model, output_dir)
    print("Done")







