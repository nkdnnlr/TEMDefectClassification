import os
import sys
import argparse

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tqdm
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_WARNINGS'] = '0'

import matplotlib.pyplot as plt
from keras.models import load_model
import skimage

import utils.helpers as helpers
from utils.predict_class import predict
from utils.eigenfiltering import eigenfiltering
from utils.preprocessing import preprocess_image, normalize_intensity, cut_intensity
from bragg_filtering_fromscratch import bragg_filtering

# import src.utils.helpers as helpers
# from src.utils.predict_class import predict
# from src.utils.eigenfiltering import eigenfiltering
# from src.utils.preprocessing import preprocess_image

def run(parent_dir, model, output_dir_, TARGET_SIZE = 128):
    """
    Segmentation with Eigenfilters
    :param parent_dir:
    :param model:
    :param output_dir:
    :return:
    """
    
    # for blurring in [16, 6, 8, 10, 12, 14, 18, 20, 22, 26, 30, 40]:
    #     for filtersize in [5, 7, 9, 11]:
    # for blurring in [16]:
    #     for filtersize in [7]:
    for blurring in [16]:
        for filtersize in [7]:
            print(f"START BLURRING {blurring} FILTERSIZE {filtersize}")
            output_dir = os.path.join(output_dir_, f'b{blurring}_fs{filtersize}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
                
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


                    bragg_filtering(original)

                    exit()

                    image = skimage.filters.gaussian(original, 2) - skimage.filters.gaussian(original, 3) #Looks quite much like previous setting
                    image = np.interp(image, (image.min(), image.max()), (0, 254))
                    

                    # image = preprocess_image(original,
                    #                  lowpass_filter="gaussian", lowpass_kernel_size=5,
                    #                  highpass_filter="laplacian", highpass_kernel_size=19,
                    #                  rescale=False)
                    
                    # fig, axes = plt.subplots(ncols=3)
                    # axes[0].imshow(original, cmap='gray')
                    # axes[1].imshow(image1, cmap='gray')
                    # axes[2].imshow(image, cmap='gray')
                    # plt.show()
                    
                    # continue
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
                    np.save(os.path.join(output_dir, name + "bestpatch.npy"), best_patch)
                    # continue


                    # # GET ANNOTADED IMAGE (WITH BEST PATCH)
                    # image_annotated = add_bestpatch_to_img(image, patchsize=128, patch=best_node, color='gray')
                    # score_map_annotated = add_bestpatch_to_img(score_map, patchsize=128, patch=best_node, color='viridis')

                    # GET LOCAL VARIANCE OF BEST PATCH
                    localvariance_patch = helpers.localvariance_filter(image=best_patch)
                    minlocalvariance = np.min(localvariance_patch)

                    # FILTER FOR LOCAL VARIANCE
                    localvariance = helpers.localvariance_filter(image=image)

                    # SAVE EVERYTHING BEFORE SYMMETRY FILTERING
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

                    np.save(os.path.join(output_dir, name + "_08_localvariance.npy"), localvariance)
                    plt.imsave(os.path.join(output_dir, name + "_08_localvariance.svg"), localvariance, cmap='inferno_r',
                               vmin=minlocalvariance - 300, vmax=minlocalvariance)

                    # continue

                    # SEGMENTING: GET BEST PATCH AND USE IT AS AN EIGENFILTER ON THE IMAGE
                    startingtime = time.time()
                    filtered = eigenfiltering(image_def=image, patch_good=best_patch, output_path=os.path.join(output_dir,
                                                                                                            'eigenfilters', name), blurring=blurring, filter_size=filtersize)
                    maxfiltered = np.max(filtered)
                    minval = np.min(filtered[filtered > 0])


                    plt.imshow(filtered, vmin=minval)
                    plt.show()
                    plt.close()




                    np.save(os.path.join(output_dir, name + "symmetry.npy"), filtered)
                    plt.imsave(os.path.join(output_dir, name + "symmetry.svg"), filtered, cmap='viridis', vmin=minval)
                               # vmax=np.max((22, maxfiltered)))


                    continue
                    
                    # print(filtered)
                    # exit()
                    
                    # exit()
                    # plt.close('all')
                    # continue

                    # filtered_normalized = normalize_intensity(cut_intensity(filtered, min=7, max=22))
                    
                    # plt.imshow(filtered_normalized)
                    # plt.show()
                    # exit()
                    # filtered_normalized = normalize_intensity(filtered)
                    # print(filtered)

                    # SAVE PLOTS

                    plt.imsave(os.path.join(output_dir, name + "_06_best_patch.svg"), best_patch, cmap='gray')
                    plt.imsave(os.path.join(output_dir, name + "_07_segmented.svg"), filtered, cmap='viridis', vmin=7,
                            vmax=np.max((22, maxfiltered)))
                    plt.imsave(os.path.join(output_dir, name + "_07_segmented_normalized.svg"), filtered, cmap='viridis', vmin=7,
                            vmax=np.max((22, maxfiltered)))
                    plt.imsave(os.path.join(output_dir, name + "_08_variance.svg"), localvariance, cmap='inferno_r',
                            vmin=minlocalvariance-300, vmax=minlocalvariance)

                    helpers.make_publication_subplots(original, score_map, filtered, localvariance, 128,
                                            best_node, maxfiltered, minlocalvariance,
                                            output_path=os.path.join(output_dir, name + "_09_all.png"))
                    plt.close('all')

            print("sum_all: ", sum_all)
            print("sum_defective: ", sum_defective)
            print("fraction_defective: ", 1.-sum_defective/sum_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_dir", "--image_dir", type=str, default="cubic/defective/images",
                        help="Name of image directory")
    # parser.add_argument("-image_dir", "--image_dir", type=str, default="12",
    #                     help="Name of image directory")
    # parser.add_argument("-image_dir", "--image_dir", type=str, default="test2",
    #                     help="Name of image directory")
    parser.add_argument("-model", "--model", type=str, default="20191208-014141.h5",
                        help="Name of model file.")
    # parser.add_argument("-output_dir", "--output_dir", type=str, default="eigenfilter_segmentation_forpost",
    #                     help="Name of model file.")
    parser.add_argument("-output_dir", "--output_dir", type=str, default="braggfiltering",
                        help="Name of model file.")
    args = parser.parse_args()

    image_dir = os.path.join('data', args.image_dir)
    assert os.path.exists(image_dir)

    output_dir = os.path.join('output', args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join('models', args.model)
    assert os.path.exists(model_path)
    model = load_model(model_path)

    run(image_dir, model, output_dir)
    print("Done")







